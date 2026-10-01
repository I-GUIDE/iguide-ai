from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_runtime.executor_factory import (
    AgentInvocationError,
    format_agent_diagnostics,
    invoke_agent_with_payload_fallback,
)


class GraphRecursionError(RuntimeError):
    pass


class _FakeExecutor:
    def invoke(self, payload, config=None):
        raise GraphRecursionError("Recursion limit of 25 reached without hitting a stop condition.")

    def get_state(self, config=None):
        messages = [
            SimpleNamespace(content="user request"),
            SimpleNamespace(
                content="",
                tool_calls=[
                    {
                        "name": "load_skill",
                        "args": {"skill_name": "chicago-crime-analysis"},
                        "id": "call_1",
                    }
                ],
            ),
            SimpleNamespace(
                content='{"status": "ok"}',
                name="load_skill",
                tool_call_id="call_1",
            ),
        ]
        return SimpleNamespace(values={"messages": messages})


def test_recursion_error_includes_tool_diagnostics():
    with pytest.raises(AgentInvocationError) as err:
        invoke_agent_with_payload_fallback(
            _FakeExecutor(),
            query="test",
            chat_history=None,
            config={"configurable": {"thread_id": "diagnostic-thread"}},
        )

    diagnostics = err.value.diagnostics
    assert diagnostics["thread_id"] == "diagnostic-thread"
    assert diagnostics["message_count"] == 3
    assert diagnostics["tool_call_counts"] == {"load_skill": 1}
    assert diagnostics["tool_calls"] == [
        {
            "name": "load_skill",
            "args": {"skill_name": "chicago-crime-analysis"},
            # The call id is carried so a result can be paired to the call that produced it
            # rather than by position within a tool name.
            "id": "call_1",
        }
    ]
    assert diagnostics["tool_results"][0]["name"] == "load_skill"
    assert "LLM tool decision: load_skill" in diagnostics["readable_trace"]
    assert "Tool result load_skill" in diagnostics["readable_trace"]


def test_format_agent_diagnostics_renders_readable_timeline():
    text = format_agent_diagnostics(
        {
            "thread_id": "thread-1",
            "recursion_limit": 60,
            "tool_call_counts": {"mcp_count_crimes_per_community": 2},
            "recent_messages": [
                {"role": "human", "content": "Which community has the most theft?"},
                {
                    "role": "ai",
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "mcp_count_crimes_per_community",
                            "args": {"crime_type": "THEFT"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "mcp_count_crimes_per_community",
                    "content": '{"top_communities": [{"name": "Loop", "count": 10}]}',
                },
                {"role": "ai", "content": "The top community is Loop."},
            ],
        }
    )

    assert "Stopped before final answer" in text
    assert "Tool calls observed: mcp_count_crimes_per_community x2." in text
    assert "User: Which community has the most theft?" in text
    assert 'LLM tool decision: mcp_count_crimes_per_community({"crime_type": "THEFT"})' in text
    assert "Tool result mcp_count_crimes_per_community" in text
    assert "LLM message: The top community is Loop." in text
