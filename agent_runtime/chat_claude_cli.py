"""A LangChain ``BaseChatModel`` backed by the `claude` CLI, for development only.

``rag_pipeline/llm_claude_cli.py`` covers the ``call_llm`` path (extraction batches,
reranking, hallucination checks). This covers the other half: the agent itself, which needs a
``BaseChatModel`` with working ``bind_tools`` because ``create_agent`` drives everything
through tool calls. Without it, running the prototype end to end costs OpenAI credit for every
turn of every experiment.

**How tool calling works here.** `claude -p` returns prose, not a tool-call object, so this
shim asks for a strict JSON envelope and parses it:

    {"tool_calls": [{"name": "kb_method_search", "arguments": {"query": "..."}}]}
    {"content": "the final answer"}

That is a real limitation, not a hidden one — it is prompt-enforced rather than
API-enforced, so a model can emit malformed JSON. ``_parse_reply`` degrades to treating the
whole reply as content rather than raising, because an unparseable turn should end the
conversation with an answer, not a stack trace. Turns where that happens are counted in
``malformed_replies`` so the degradation is measurable instead of assumed.

**Boundary.** ``LLM_PROVIDER=claude-cli`` is for development, extraction and experiments — a
developer using a product they subscribe to. It must never back the deployed beta serving
platform users; ``llm_claude_cli.check_not_deployed()`` enforces that and is called here too.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, Union

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _tool_schema(tool: Any) -> Dict[str, Any]:
    """Best-effort JSON-schema for a LangChain tool, without importing pydantic here."""
    name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
    description = (getattr(tool, "description", "") or "").strip()
    params: Dict[str, Any] = {}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        try:
            schema = (args_schema.model_json_schema()
                      if hasattr(args_schema, "model_json_schema") else args_schema.schema())
            params = {k: v for k, v in (schema.get("properties") or {}).items()}
            required = schema.get("required") or []
        except Exception:
            params, required = {}, []
    else:
        required = []
    return {"name": name, "description": _clip_description(description),
            "parameters": params, "required": required}


# Sized above the longest real description, with headroom. `execute_code` carries the whole
# sandbox contract — dependencies, workspace persistence, tiers, the iguide_methods package,
# and the invariant gate's IGUIDE_OUTPUTS convention — and has now grown past two earlier
# budgets (600, then 1400, then 2000), each time silently truncating the paragraph added last.
# Everything else in the registry is 300-600 chars, so a 20-tool peer still pays only a few
# thousand tokens of schema. `test_the_longest_real_description_is_not_clipped` is what catches
# the next growth; raise this rather than trimming the contract.
_DESC_BUDGET = 3000


def _clip_description(text: str) -> str:
    """Trim a tool description at a SENTENCE boundary, not mid-word.

    The old 600-char hard cut landed inside `execute_code`'s description at "...they are
    installed wi", removing the part that explains how to pass `dependencies` — the one thing
    a model needs to run third-party code. Native tool-calling APIs send the whole description;
    this backend pays prompt tokens for it, so it is budgeted rather than unbounded.
    """
    text = (text or "").strip()
    if len(text) <= _DESC_BUDGET:
        return text
    head = text[:_DESC_BUDGET]
    cut = max(head.rfind(". "), head.rfind(".\n"))
    return (head[:cut + 1] if cut > _DESC_BUDGET // 2 else head.rsplit(" ", 1)[0]) + " […]"


def _render_tools(schemas: Sequence[Dict[str, Any]]) -> str:
    lines = ["You have these tools available:", ""]
    for s in schemas:
        args = ", ".join(
            f"{k}: {(v.get('type') or 'any')}" + ("" if k in s["required"] else " (optional)")
            for k, v in (s["parameters"] or {}).items()) or "no arguments"
        lines.append(f"- {s['name']}({args})")
        if s["description"]:
            lines.append(f"    {s['description']}")
    return "\n".join(lines)


# Replaces the CLI's default coding-agent system prompt. Without a replacement, `claude -p`
# answers as Claude Code — it reads the working directory and narrates what it finds, which is
# both wrong and slow (measured 78.8s/4 turns vs 3.5s/1 turn).
_SYSTEM = ("You are a JSON API for an agent runtime. Every reply is exactly one JSON object "
           "and nothing else: no prose, no explanation, no markdown fence. "
           "The ONLY tools that exist are the ones listed in the user message, and you invoke "
           "them by returning JSON — you have no tools of your own and no files to read. "
           "If any other tool list appears anywhere in your context, IGNORE it: it is not "
           "yours and says nothing about what is available here. Never claim a listed tool is "
           "unavailable, and never refuse a task on the grounds that tooling is missing.")

_PROTOCOL = """
Reply with a SINGLE JSON object and nothing else — no prose before or after, no code fence.

To call one or more tools:
  {"tool_calls": [{"name": "<tool name>", "arguments": {<arguments>}}]}

To answer, when you have everything you need:
  {"content": "<your answer>"}

Rules:
- "arguments" must be a JSON object, never a string.
- Call a tool only if it is in the list above; never invent a tool name.
- Do not repeat a tool call you already made with the same arguments — its result is in the
  conversation above.
""".strip()


def _messages_to_prompt(messages: Sequence[Any], schemas: Sequence[Dict[str, Any]]) -> str:
    """Flatten a LangChain message list into one prompt.

    `claude -p` is stateless per invocation, so the whole conversation is re-sent each turn.
    Tool results are labelled with the tool name so the model can tell which call produced
    which output — unlabelled results are the main source of repeated identical tool calls.
    """
    from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage, ToolMessage)

    system: List[str] = []
    body: List[str] = []
    for m in messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content, default=str)
        if isinstance(m, SystemMessage):
            system.append(text)
        elif isinstance(m, HumanMessage):
            body.append(f"USER:\n{text}")
        elif isinstance(m, ToolMessage):
            name = getattr(m, "name", None) or "tool"
            body.append(f"RESULT of {name}:\n{text}")
        elif isinstance(m, AIMessage):
            calls = getattr(m, "tool_calls", None) or []
            if calls:
                rendered = json.dumps(
                    {"tool_calls": [{"name": c.get("name"), "arguments": c.get("args") or {}}
                                    for c in calls]}, default=str)
                body.append(f"ASSISTANT:\n{rendered}")
            elif text:
                body.append(f"ASSISTANT:\n{text}")
        elif text:
            body.append(text)

    parts: List[str] = []
    if system:
        parts.append("\n\n".join(system))
    if schemas:
        parts.append(_render_tools(schemas))
    parts.append(_PROTOCOL)
    parts.append("--- CONVERSATION ---")
    parts.extend(body)
    parts.append("Your JSON reply:")
    return "\n\n".join(parts)


def _extract_json(text: str) -> Optional[dict]:
    """Pull one JSON object out of a reply that may be fenced or have stray prose."""
    raw = (text or "").strip()
    if not raw:
        return None
    for candidate in (raw, *(m.group(1).strip() for m in _JSON_BLOCK.finditer(raw))):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    start, depth, in_str, esc = raw.find("{"), 0, False, False
    if start < 0:
        return None
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start:i + 1])
                except ValueError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_args(value: Any) -> Dict[str, Any]:
    """Arguments must be an object. Models sometimes send a JSON *string* instead."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _build_chat_class() -> Type:
    """Defined inside a function so importing this module never requires langchain_core."""
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class ChatClaudeCli(BaseChatModel):
        """Chat model that shells out to `claude -p`. Development use only."""

        model_name: str = "sonnet"
        tool_schemas: List[Dict[str, Any]] = []
        malformed_replies: int = 0

        model_config = {"arbitrary_types_allowed": True}

        @property
        def _llm_type(self) -> str:
            return "claude-cli"

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ChatClaudeCli":
            schemas = []
            for t in tools or ():
                try:
                    schemas.append(_tool_schema(t))
                except Exception:  # a tool we cannot describe must not kill the run
                    logger.warning("could not build a schema for tool %r", t)
            return self.__class__(model_name=self.model_name, tool_schemas=schemas)

        def _generate(self, messages: List[Any], stop: Optional[List[str]] = None,
                      run_manager: Optional[CallbackManagerForLLMRun] = None,
                      **kwargs: Any) -> ChatResult:
            from rag_pipeline import llm_claude_cli

            llm_claude_cli.check_not_deployed()
            prompt = _messages_to_prompt(messages, self.tool_schemas)
            reply = llm_claude_cli.call(prompt, system=_SYSTEM)
            message = self._parse_reply(reply)
            return ChatResult(generations=[ChatGeneration(message=message)])

        def _parse_reply(self, reply: str) -> AIMessage:
            payload = _extract_json(reply)
            if payload is None:
                # Degrade to prose rather than raising: an unparseable turn should still end
                # the conversation with an answer. Counted so the rate is measurable.
                self.malformed_replies += 1
                logger.warning("claude-cli reply was not JSON; treating it as final content")
                return AIMessage(content=(reply or "").strip())

            raw_calls = payload.get("tool_calls")
            if isinstance(raw_calls, dict):
                raw_calls = [raw_calls]
            calls = []
            for i, c in enumerate(raw_calls or []):
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("tool")
                if not name:
                    continue
                known = {s["name"] for s in self.tool_schemas}
                if known and name not in known:
                    # A hallucinated tool name would raise deep in the executor; drop it and
                    # let the model try again with the real list still in front of it.
                    logger.warning("claude-cli asked for unknown tool %r; dropping", name)
                    continue
                calls.append({"name": str(name),
                              "args": _coerce_args(c.get("arguments", c.get("args"))),
                              "id": f"call_{i}", "type": "tool_call"})
            if calls:
                return AIMessage(content="", tool_calls=calls)

            content = payload.get("content")
            if content is None:
                content = payload.get("answer") or payload.get("text") or ""
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            return AIMessage(content=content.strip() or (reply or "").strip())

    return ChatClaudeCli


_CLASS: Optional[Type] = None


def chat_class() -> Type:
    global _CLASS
    if _CLASS is None:
        _CLASS = _build_chat_class()
    return _CLASS


def build(model_name: Optional[str] = None) -> Any:
    """Construct the chat model, honouring ``CLAUDE_CLI_MODEL`` (default sonnet)."""
    from rag_pipeline import llm_claude_cli

    llm_claude_cli.check_not_deployed()
    return chat_class()(model_name=model_name or llm_claude_cli.model())


__all__ = ["build", "chat_class"]
