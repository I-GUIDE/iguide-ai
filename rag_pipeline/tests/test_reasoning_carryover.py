"""A reasoning model's chain-of-thought must survive into the next step.

AnvilGPT's gpt-oss:120b answers a tool-calling step with `content: None`,
`reasoning_content: "<the plan>"` and the tool call. langchain_openai 1.6 keeps neither
reasoning field, so the replayed assistant turn is a bare tool call with no rationale; the
model re-derives from the user request and re-issues the same call. Observed live:
admin_boundary five times in one turn with near-identical args.

These pin the raw payload shape as the provider actually sends it.
"""
from __future__ import annotations

import pytest

from agent_runtime import executor_factory as ef


def _raw(content, reasoning, *, tool_calls=True):
    """A chat/completions payload in the shape AnvilGPT returns for gpt-oss:120b."""
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [{
            "id": "chatcmpl-tool-8b6c9becd846dda5", "type": "function",
            "function": {"name": "admin_boundary",
                         "arguments": '{"area": "Champaign County", "state": "Illinois"}'}}]
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-oss:120b",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


@pytest.fixture()
def llm():
    cls = pytest.importorskip("agent_runtime.executor_factory") and ef._reasoning_preserving_chat_openai()
    return cls(model="gpt-oss:120b", api_key="test-key", base_url="http://x/api")


PLAN = ("The user asks for census tracts of Champaign County. I should call admin_boundary "
        "with subdivide=tracts, then embed the returned zones.")


def test_reasoning_is_promoted_into_content_on_a_tool_calling_step(llm):
    msg = llm._create_chat_result(_raw(None, PLAN)).generations[0].message
    assert msg.content == PLAN, "the plan must round-trip as ordinary assistant content"
    assert msg.additional_kwargs["reasoning_content"] == PLAN
    assert [c["name"] for c in msg.tool_calls] == ["admin_boundary"]


def test_it_survives_serialization_back_to_the_provider(llm):
    """The whole point: what gets SENT on the next step must carry the rationale."""
    from langchain_openai.chat_models.base import _convert_message_to_dict

    msg = llm._create_chat_result(_raw(None, PLAN)).generations[0].message
    sent = _convert_message_to_dict(msg)
    assert PLAN in str(sent.get("content")), sent


def test_a_final_message_is_left_alone(llm):
    """Promoting reasoning onto a message with no tool call would leak it into the answer."""
    msg = llm._create_chat_result(_raw(None, PLAN, tool_calls=False)).generations[0].message
    assert not (msg.content or "").strip()
    assert msg.additional_kwargs["reasoning_content"] == PLAN


def test_real_content_is_never_overwritten(llm):
    msg = llm._create_chat_result(_raw("Here are the tracts.", PLAN)).generations[0].message
    assert msg.content == "Here are the tracts."


def test_a_provider_without_reasoning_is_untouched(llm):
    msg = llm._create_chat_result(_raw(None, None)).generations[0].message
    assert "reasoning_content" not in msg.additional_kwargs


def test_a_very_long_chain_of_thought_is_capped_keeping_the_conclusion(llm):
    long_plan = "x" * 9000 + " THEREFORE call admin_boundary once."
    msg = llm._create_chat_result(_raw(None, long_plan)).generations[0].message
    assert len(msg.content) == ef._REASONING_CARRYOVER_MAX
    assert msg.content.endswith("THEREFORE call admin_boundary once."), "kept the wrong end"


def test_a_provider_quirk_never_breaks_the_turn(llm, monkeypatch):
    """If our carry-over layer throws, the turn still gets the base class's result."""
    monkeypatch.setattr(ef, "_reasoning_text",
                        lambda choice: (_ for _ in ()).throw(ValueError("boom")))
    msg = llm._create_chat_result(_raw(None, PLAN)).generations[0].message
    assert [c["name"] for c in msg.tool_calls] == ["admin_boundary"], "lost the tool call"


def test_both_construction_paths_use_the_subclass():
    """The model PICKER path is where gpt-oss is actually selected — pin both."""
    import inspect
    src = inspect.getsource(ef)
    assert src.count("_reasoning_preserving_chat_openai()") >= 2
