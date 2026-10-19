"""LangChain face of the agent's own model client (architecture stage 88).

:class:`OwnChatModel` is a ``BaseChatModel`` over :class:`agent_runtime.model_client.ChatClient`.
It takes the constructor arguments the agent passes ``ChatOpenAI`` (``model``, ``api_key``,
``base_url``, ``temperature``, ``reasoning_effort``) and sends the same ``/chat/completions``
body, so ``build_default_llm`` / ``build_llm`` callers, ``create_agent`` and the LangGraph
supervisor do not change. ``AGENT_MODEL_CLIENT=own`` selects it.

What stays exactly as it was, and why it matters:

* **The wire body.** Messages are converted the way langchain_openai converts them (a test
  compares the two on every message shape the agent sends), tools the way ``bind_tools`` formats
  them, and the same parameters are sent or left out. Harness runs on the two clients compare
  the client, not a changed prompt.
* **The reasoning carry-over.** A provider's ``reasoning_content`` / ``reasoning`` / ``thinking``
  is kept on ``additional_kwargs`` and, on a TOOL-CALLING step whose ``content`` is empty,
  promoted into ``content`` (its tail), as ``ReasoningPreservingChatOpenAI`` does. It is never
  sent back as a field: deepseek rejects ``reasoning_content`` in input messages.
* **Cost accounting.** ``usage_metadata`` is built as langchain_openai builds it, and the
  invocation params carry ``model`` and ``base_url``, which is all ``UsageCallbackHandler``
  reads to price a call and to recognise Lumen's (stage 59).

What changes is underneath: a deadline per attempt, Stop honoured mid-call, bounded retries
(``model_client``), and a provider error that keeps the provider's own message.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import InputTokenDetails, OutputTokenDetails, UsageMetadata
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import make_invalid_tool_call, parse_tool_call
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from agent_runtime.model_client import ChatClient

DEFAULT_BASE_URL = "https://api.openai.com/v1"

# The same keys and cap as executor_factory's carry-over: one behaviour, two clients.
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")
_REASONING_CARRYOVER_MAX = 4000
# Content blocks a chat/completions request must not carry (langchain_openai drops the same).
_DROP_BLOCKS = ("tool_use", "thinking", "reasoning_content", "function_call",
                "code_interpreter_call")


# --- outbound: LangChain messages -> chat/completions dicts --------------------------------------
def _format_content(content: Any) -> Any:
    if not (content and isinstance(content, list)):
        return content
    from langchain_core.messages import convert_to_openai_data_block, is_data_content_block

    out: List[Any] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in _DROP_BLOCKS:
            continue
        if isinstance(block, dict) and is_data_content_block(block):
            out.append(convert_to_openai_data_block(block))
        elif (isinstance(block, dict) and block.get("type") == "image"
              and isinstance(block.get("source"), dict)):
            src = block["source"]
            if src.get("type") == "base64" and src.get("media_type") and src.get("data"):
                out.append({"type": "image_url", "image_url": {
                    "url": f"data:{src['media_type']};base64,{src['data']}"}})
            elif src.get("type") == "url" and src.get("url"):
                out.append({"type": "image_url", "image_url": {"url": src["url"]}})
        else:
            out.append(block)
    return out


def _ai_content(message: AIMessage) -> Any:
    """v1 content blocks back to chat/completions: text kept, reasoning/tool_call blocks dropped."""
    if not isinstance(message.content, list):
        return message.content
    out: List[Any] = []
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append({"type": "text", "text": block.get("text", "")})
        elif isinstance(block, dict) and block.get("type") in ("reasoning", "tool_call"):
            continue
        else:
            out.append(block)
    return out


def message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """One message as chat/completions expects it (langchain_openai's rules, pinned by a test)."""
    raw = _ai_content(message) if isinstance(message, AIMessage) else message.content
    out: Dict[str, Any] = {"content": _format_content(raw)}
    name = message.name or message.additional_kwargs.get("name")
    if name is not None:
        out["name"] = name
    if isinstance(message, ChatMessage):
        out["role"] = message.role
    elif isinstance(message, HumanMessage):
        out["role"] = "user"
    elif isinstance(message, AIMessage):
        out["role"] = "assistant"
        if message.tool_calls or message.invalid_tool_calls:
            out["tool_calls"] = [
                {"type": "function", "id": tc["id"],
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for tc in message.tool_calls
            ] + [
                {"type": "function", "id": tc["id"],
                 "function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            out["tool_calls"] = [
                {k: v for k, v in tc.items() if k in ("id", "type", "function")}
                for tc in message.additional_kwargs["tool_calls"]
            ]
        elif "function_call" in message.additional_kwargs:
            out["function_call"] = message.additional_kwargs["function_call"]
        if "tool_calls" in out or "function_call" in out:
            out["content"] = out["content"] or None
    elif isinstance(message, SystemMessage):
        out["role"] = message.additional_kwargs.get("__openai_role__", "system")
    elif isinstance(message, FunctionMessage):
        out["role"] = "function"
    elif isinstance(message, ToolMessage):
        out = {"content": out["content"], "role": "tool", "tool_call_id": message.tool_call_id}
    else:
        raise TypeError(f"Got unknown type {message}")
    return out


# --- inbound: chat/completions dicts -> LangChain messages ---------------------------------------
def _reasoning_text(message: Mapping[str, Any]) -> str:
    for key in _REASONING_KEYS:
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def usage_metadata(usage: Mapping[str, Any]) -> UsageMetadata:
    """The provider's usage, shaped as langchain_openai shapes it (cost accounting reads it)."""
    tokens_in = usage.get("prompt_tokens") or 0
    tokens_out = usage.get("completion_tokens") or 0
    details_in = {"audio": (usage.get("prompt_tokens_details") or {}).get("audio_tokens"),
                  "cache_read": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}
    details_out = {"audio": (usage.get("completion_tokens_details") or {}).get("audio_tokens"),
                   "reasoning": (usage.get("completion_tokens_details") or {}).get(
                       "reasoning_tokens")}
    return UsageMetadata(
        input_tokens=tokens_in, output_tokens=tokens_out,
        total_tokens=usage.get("total_tokens") or tokens_in + tokens_out,
        input_token_details=InputTokenDetails(**{k: v for k, v in details_in.items()
                                                 if v is not None}),
        output_token_details=OutputTokenDetails(**{k: v for k, v in details_out.items()
                                                   if v is not None}))


def dict_to_ai_message(message: Mapping[str, Any]) -> AIMessage:
    """An assistant reply, with the reasoning carry-over applied."""
    content = message.get("content") or ""
    kwargs: Dict[str, Any] = {}
    if message.get("function_call"):
        kwargs["function_call"] = dict(message["function_call"])
    if message.get("audio"):
        kwargs["audio"] = message["audio"]
    if "refusal" in message:
        kwargs["refusal"] = message.get("refusal")
    calls, invalid = [], []
    for raw in message.get("tool_calls") or []:
        try:
            calls.append(parse_tool_call(raw, return_id=True))
        except Exception as exc:  # noqa: BLE001 - kept as an invalid call, as langchain does
            invalid.append(make_invalid_tool_call(raw, str(exc)))
    reasoning = _reasoning_text(message)
    if reasoning:
        kwargs["reasoning_content"] = reasoning
        if calls and not (isinstance(content, str) and content.strip()):
            # Keep the TAIL: the decision the model reached sits at the end.
            content = reasoning[-_REASONING_CARRYOVER_MAX:]
    return AIMessage(content=content, additional_kwargs=kwargs, name=message.get("name"),
                     id=message.get("id"), tool_calls=calls, invalid_tool_calls=invalid)


class OwnChatModel(BaseChatModel):
    """``ChatOpenAI``'s constructor and wire format over the agent's own client."""

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

    model_name: str = Field(default="gpt-4o-2024-11-20", alias="model")
    api_key: Optional[SecretStr] = Field(default=None, exclude=True, repr=False)
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    # Per-attempt deadline and retries; None reads AGENT_LLM_TIMEOUT_SECONDS / _MAX_RETRIES.
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    streaming: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    extra_body: Optional[Dict[str, Any]] = None
    default_headers: Optional[Dict[str, str]] = None

    _client: Optional[ChatClient] = PrivateAttr(default=None)

    # -- identity ----------------------------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "own-openai-compatible"

    @property
    def model(self) -> str:
        return self.model_name

    @property
    def resolved_base_url(self) -> str:
        import os

        return (self.base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        # model and base_url are what UsageCallbackHandler reads (the key never is).
        return {"model": self.model_name, "model_name": self.model_name,
                "base_url": self.resolved_base_url, **self._default_params}

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "openai"
        params["ls_model_name"] = self.model_name
        if self.temperature is not None:
            params["ls_temperature"] = self.temperature
        return params

    @property
    def client(self) -> ChatClient:
        if self._client is None:
            import os

            key = (self.api_key.get_secret_value() if self.api_key is not None
                   else os.getenv("OPENAI_API_KEY"))
            self._client = ChatClient(self.resolved_base_url, key, timeout=self.timeout,
                                      max_retries=self.max_retries,
                                      headers=self.default_headers)
        return self._client

    # -- the request -------------------------------------------------------------------------
    @property
    def _default_params(self) -> Dict[str, Any]:
        optional = {"stop": self.stop or None, "max_tokens": self.max_tokens,
                    "temperature": self.temperature, "reasoning_effort": self.reasoning_effort}
        return {"model": self.model_name, "stream": self.streaming,
                **{k: v for k, v in optional.items() if v is not None}, **self.model_kwargs}

    def _body(self, messages: List[BaseMessage], stop: Optional[List[str]],
              kwargs: Dict[str, Any]) -> Dict[str, Any]:
        body = {**self._default_params, **kwargs}
        if stop is not None:
            body["stop"] = stop
        extra = body.pop("extra_body", None) or self.extra_body
        if extra:
            body.update(extra)
        body["messages"] = [message_to_dict(m) for m in messages]
        return body

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None,
                   strict: Optional[bool] = None, parallel_tool_calls: Optional[bool] = None,
                   **kwargs: Any) -> Any:
        """``ChatOpenAI.bind_tools``'s formatting and tool_choice rules."""
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        formatted = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        names = [t["function"]["name"] for t in formatted if "function" in t]
        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice in names:
                    tool_choice = {"type": "function", "function": {"name": tool_choice}}
                elif tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif not isinstance(tool_choice, dict):
                raise ValueError(f"Unrecognized tool_choice type: {tool_choice!r}")
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)

    # -- calls -------------------------------------------------------------------------------
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        if self.streaming:
            from langchain_core.language_models.chat_models import generate_from_stream

            return generate_from_stream(self._stream(messages, stop, run_manager, **kwargs))
        timeout = kwargs.pop("timeout", None)
        response = self.client.complete(self._body(messages, stop, kwargs), timeout=timeout)
        return self._chat_result(response)

    def _chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        if response.get("error"):
            raise ValueError(response.get("error"))
        choices = response.get("choices")
        if choices is None:
            raise TypeError("Received response with null value for `choices`."
                            if "choices" in response
                            else f"Response missing `choices` key: {list(response)}")
        usage = response.get("usage")
        generations = []
        for choice in choices:
            message = dict_to_ai_message(choice.get("message") or {})
            if usage:
                message.usage_metadata = usage_metadata(usage)
            info: Dict[str, Any] = {"finish_reason": choice.get("finish_reason")}
            if "logprobs" in choice:
                info["logprobs"] = choice["logprobs"]
            generations.append(ChatGeneration(message=message, generation_info=info))
        llm_output: Dict[str, Any] = {
            "token_usage": usage, "model_provider": "openai",
            "model_name": response.get("model", self.model_name),
            "system_fingerprint": response.get("system_fingerprint", "")}
        if "id" in response:
            llm_output["id"] = response["id"]
        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        timeout = kwargs.pop("timeout", None)
        body = self._body(messages, stop, kwargs)
        reasoning: List[str] = []
        saw_text = False
        saw_calls = False
        for chunk in self.client.stream(body, timeout=timeout):
            usage = chunk.get("usage")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for key in _REASONING_KEYS:
                    if isinstance(delta.get(key), str) and delta[key]:
                        reasoning.append(delta[key])
                        break
                text = delta.get("content") or ""
                saw_text = saw_text or bool(text.strip())
                calls = []
                for raw in delta.get("tool_calls") or []:
                    fn = raw.get("function") or {}
                    calls.append(tool_call_chunk(name=fn.get("name"), args=fn.get("arguments"),
                                                 id=raw.get("id"), index=raw.get("index")))
                saw_calls = saw_calls or bool(calls)
                info = ({"finish_reason": choice["finish_reason"]}
                        if choice.get("finish_reason") else None)
                if text or calls or info:
                    msg = AIMessageChunk(content=text, tool_call_chunks=calls)
                    gen = ChatGenerationChunk(message=msg, generation_info=info)
                    if run_manager and text:
                        run_manager.on_llm_new_token(text, chunk=gen)
                    yield gen
            if usage:
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="", usage_metadata=usage_metadata(usage)))
        thought = "".join(reasoning).strip()
        if thought:
            # The carry-over, at the end of the stream, when it is known whether this was a
            # tool-calling step with no text of its own.
            promote = saw_calls and not saw_text
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=thought[-_REASONING_CARRYOVER_MAX:] if promote else "",
                additional_kwargs={"reasoning_content": thought}))


__all__ = ["OwnChatModel", "dict_to_ai_message", "message_to_dict", "usage_metadata"]
