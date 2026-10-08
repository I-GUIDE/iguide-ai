"""The `claude` CLI as the agent's chat model, including tool calling.

``llm_claude_cli`` covered ``call_llm``; the agent itself needs a ``BaseChatModel`` with
working ``bind_tools`` because ``create_agent`` drives everything through tool calls. Without
this every agent turn of every experiment billed OpenAI even with LLM_PROVIDER=claude-cli.

Tool calling is prompt-enforced, not API-enforced, so the parser is the load-bearing part and
gets most of the attention here: a malformed reply must degrade to an answer rather than
raise, and a hallucinated tool name must be dropped rather than reach the executor.

No subprocess is started — ``llm_claude_cli.call`` is monkeypatched — so these are pure.
"""

from __future__ import annotations

import json

import pytest

from agent_runtime import chat_claude_cli as ccc


@pytest.fixture()
def model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    replies = {"text": ""}

    def fake_call(prompt, *, system=None):
        replies["prompt"] = prompt
        replies["system"] = system
        return replies["text"]

    from rag_pipeline import llm_claude_cli
    monkeypatch.setattr(llm_claude_cli, "call", fake_call)
    monkeypatch.setattr(llm_claude_cli, "check_not_deployed", lambda: None)
    m = ccc.chat_class()(model_name="sonnet",
                         tool_schemas=[{"name": "kb_method_search", "description": "find methods",
                                        "parameters": {"query": {"type": "string"}},
                                        "required": ["query"]}])
    return m, replies


def _invoke(model_and_replies, reply_text):
    m, replies = model_and_replies
    replies["text"] = reply_text
    from langchain_core.messages import HumanMessage
    return m.invoke([HumanMessage(content="find something")])


# ------------------------------------------------------------------ tool calls

def test_a_json_tool_call_becomes_a_real_tool_call(model):
    out = _invoke(model, '{"tool_calls":[{"name":"kb_method_search","arguments":{"query":"crime"}}]}')
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["name"] == "kb_method_search"
    assert out.tool_calls[0]["args"] == {"query": "crime"}


def test_a_fenced_reply_is_still_parsed():
    """Models wrap JSON in a code fence constantly, however firmly they are told not to."""
    assert ccc._extract_json('```json\n{"content":"hi"}\n```') == {"content": "hi"}


def test_json_with_prose_around_it_is_recovered():
    assert ccc._extract_json('Sure! {"content":"hi"} Hope that helps.') == {"content": "hi"}


def test_arguments_sent_as_a_json_string_are_coerced(model):
    """A recurring model habit: "arguments": "{\\"query\\": \\"x\\"}" instead of an object."""
    out = _invoke(model, '{"tool_calls":[{"name":"kb_method_search","arguments":"{\\"query\\":\\"x\\"}"}]}')
    assert out.tool_calls[0]["args"] == {"query": "x"}


def test_a_single_tool_call_object_is_accepted_like_a_list(model):
    out = _invoke(model, '{"tool_calls":{"name":"kb_method_search","arguments":{"query":"x"}}}')
    assert len(out.tool_calls) == 1


def test_a_hallucinated_tool_name_is_dropped_not_forwarded(model):
    """An unknown name would raise deep inside the executor; drop it and let the model retry."""
    out = _invoke(model, '{"tool_calls":[{"name":"definitely_not_a_tool","arguments":{}}]}')
    assert out.tool_calls == []


def test_a_content_reply_carries_no_tool_calls(model):
    out = _invoke(model, '{"content":"the answer"}')
    assert out.tool_calls == []
    assert out.content == "the answer"


# ------------------------------------------------------------------ degradation

def test_a_non_json_reply_becomes_content_rather_than_an_exception(model):
    m, _ = model
    out = _invoke(model, "I could not follow the protocol, but here is the answer anyway.")
    assert "answer anyway" in out.content
    assert out.tool_calls == []
    assert m.malformed_replies == 1, "the degradation must be counted, not silent"


def test_an_empty_reply_does_not_raise(model):
    out = _invoke(model, "")
    assert out.content == ""


def test_alternative_answer_keys_are_accepted(model):
    assert _invoke(model, '{"answer":"forty-two"}').content == "forty-two"


# ------------------------------------------------------------------ prompt assembly

def test_the_prompt_carries_tools_the_protocol_and_the_conversation(model):
    m, replies = model
    _invoke(model, '{"content":"ok"}')
    prompt = replies["prompt"]
    assert "kb_method_search" in prompt
    assert "tool_calls" in prompt
    assert "find something" in prompt


def test_the_system_prompt_replaces_the_coding_agent_persona(model):
    """Without this the CLI answers as Claude Code and reads the working directory:
    measured 78.8s over 4 turns, versus 3.5s over 1."""
    m, replies = model
    _invoke(model, '{"content":"ok"}')
    assert "JSON" in (replies["system"] or "")


def test_tool_results_are_labelled_with_their_tool(model):
    """Unlabelled results are the main cause of the model repeating a call it already made."""
    m, replies = model
    replies["text"] = '{"content":"ok"}'
    from langchain_core.messages import HumanMessage, ToolMessage
    m.invoke([HumanMessage(content="q"),
              ToolMessage(content="RESULTDATA", name="kb_method_search", tool_call_id="c0")])
    assert "RESULT of kb_method_search" in replies["prompt"]


def test_a_prior_tool_call_is_replayed_so_the_model_sees_what_it_asked(model):
    m, replies = model
    replies["text"] = '{"content":"ok"}'
    from langchain_core.messages import AIMessage, HumanMessage
    prior = AIMessage(content="", tool_calls=[{"name": "kb_method_search",
                                               "args": {"query": "crime"},
                                               "id": "c0", "type": "tool_call"}])
    m.invoke([HumanMessage(content="q"), prior])
    assert "kb_method_search" in replies["prompt"] and "crime" in replies["prompt"]


# ------------------------------------------------------------------ wiring

def test_bind_tools_builds_schemas_from_real_tools(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    from agent_runtime.langchain_granular_tools import make_langchain_granular_tools

    tools = [t for t in make_langchain_granular_tools() if t.name == "kb_method_search"]
    bound = ccc.chat_class()(model_name="sonnet").bind_tools(tools)
    assert [s["name"] for s in bound.tool_schemas] == ["kb_method_search"]
    assert "query" in bound.tool_schemas[0]["parameters"]


def test_the_provider_switch_selects_this_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setenv("CLAUDE_CLI_MODEL", "sonnet")
    from agent_runtime.executor_factory import build_default_llm

    assert type(build_default_llm()).__name__ == "ChatClaudeCli"


# ------------------------------------------------------------------ CLI isolation

def test_the_cli_is_invoked_with_agent_tools_denied(monkeypatch):
    """`claude -p` defaults to the full coding agent — it reads the cwd and acts. These flags
    are the subscription-compatible stand-in for --bare, which cannot read OAuth."""
    from rag_pipeline import llm_claude_cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_BARE", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_ISOLATE", raising=False)
    argv = llm_claude_cli._build_argv("claude", "prompt", "sonnet")
    assert "--disallowed-tools" in argv
    assert "--strict-mcp-config" in argv
    assert "--system-prompt" in argv
    denied = argv[argv.index("--disallowed-tools") + 1]
    assert {"Bash", "Read", "WebFetch"} <= set(denied.split(","))


def test_isolation_can_be_disabled_for_debugging(monkeypatch):
    from rag_pipeline import llm_claude_cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CLI_ISOLATE", "0")
    argv = llm_claude_cli._build_argv("claude", "prompt", "sonnet")
    assert "--disallowed-tools" not in argv


def test_a_deployment_marker_refuses_this_backend(monkeypatch):
    from rag_pipeline import llm_claude_cli

    monkeypatch.setenv("LLM_PROVIDER", "claude-cli")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    with pytest.raises(RuntimeError, match="development-only"):
        llm_claude_cli.check_not_deployed()


def test_the_longest_real_description_is_not_clipped():
    """Regression guard on the budget.

    execute_code's description carries the whole sandbox contract. A 600-char cut removed how
    to pass `dependencies`; a later 1400-char cut removed the paragraph naming the
    `iguide_methods` package. Both were silent, and both changed what the model could do.
    """
    from agent_runtime.chat_claude_cli import _tool_schema
    from agent_runtime.langchain_exec_tools import make_code_execution_tools

    tool = make_code_execution_tools()[0]
    schema = _tool_schema(tool)
    assert schema["description"] == (tool.description or "").strip(), (
        "execute_code's description is being clipped again; raise _DESC_BUDGET")


def test_a_pathological_description_is_still_bounded():
    from agent_runtime.chat_claude_cli import _clip_description, _DESC_BUDGET

    clipped = _clip_description("word " * 5000)
    assert len(clipped) <= _DESC_BUDGET + 16
    assert clipped.endswith("[…]")


def test_clipping_lands_on_a_sentence_boundary_when_it_can():
    from agent_runtime.chat_claude_cli import _clip_description, _DESC_BUDGET

    text = ("A" * (_DESC_BUDGET // 2)) + ". " + ("B" * _DESC_BUDGET) + ". tail"
    assert _clip_description(text).endswith(". […]")


def test_the_cli_is_left_with_no_tools_of_its_own(monkeypatch):
    """The CLI advertises its remaining tools to the model, and under a long prompt the model
    believes that list over the prompt's. Observed verbatim in an agent turn: "Only a limited
    set of tools (AskUserQuestion, ScheduleWakeup, ShareOnboardingGuide, Skill, and ToolSearch)
    are callable here" — exactly the tools left allowed — then a refusal to run code because
    execute_code was "not available", while it was bound and working."""
    from rag_pipeline import llm_claude_cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_BARE", raising=False)
    monkeypatch.delenv("CLAUDE_CLI_ISOLATE", raising=False)
    argv = llm_claude_cli._build_argv("claude", "prompt", "sonnet")
    denied = set(argv[argv.index("--disallowed-tools") + 1].split(","))
    assert {"AskUserQuestion", "ScheduleWakeup", "ShareOnboardingGuide", "Skill",
            "ToolSearch"} <= denied, "the CLI can still offer the model a competing tool list"


def test_the_system_prompt_disowns_any_other_tool_list():
    from agent_runtime.chat_claude_cli import _SYSTEM

    assert "IGNORE" in _SYSTEM
    assert "unavailable" in _SYSTEM


# ------------------------------------------------- the corrective retry for a missing envelope

def test_a_reply_with_no_envelope_is_retried_once(monkeypatch):
    """A prose reply ENDS the agent loop.

    `_parse_reply` degrades to treating the whole reply as the final answer, which is right for
    a last turn and wrong for every other one: the turn that was going to call `execute_code`
    becomes the turn that says "I don't have access to run that". Measured at 0-6 per run while
    A/B-ing the knowledge base, which swamped the effect under test.
    """
    from agent_runtime import chat_claude_cli
    from rag_pipeline import llm_claude_cli

    calls = []

    def fake_call(prompt, system=None):
        calls.append(prompt)
        return ("sure, I can help with that"
                if len(calls) == 1
                else '{"tool_calls": [{"name": "kb_method_search", "arguments": {"query": "x"}}]}')

    monkeypatch.setattr(llm_claude_cli, "call", fake_call)
    monkeypatch.setattr(llm_claude_cli, "check_not_deployed", lambda: None)

    model = chat_claude_cli.build()
    model.tool_schemas = [{"name": "kb_method_search", "description": "d", "parameters": {}}]
    result = model._generate([])

    assert len(calls) == 2, "no retry was attempted"
    assert "not valid JSON" in calls[1], "the retry did not restate the contract"
    assert result.generations[0].message.tool_calls, "the retried envelope was not used"


def test_a_good_reply_is_not_retried(monkeypatch):
    """The retry costs a second CLI call; it must fire only on the failure path."""
    from agent_runtime import chat_claude_cli
    from rag_pipeline import llm_claude_cli

    calls = []

    def fake_call(prompt, system=None):
        calls.append(prompt)
        return '{"content": "done"}'

    monkeypatch.setattr(llm_claude_cli, "call", fake_call)
    monkeypatch.setattr(llm_claude_cli, "check_not_deployed", lambda: None)

    model = chat_claude_cli.build()
    model.tool_schemas = [{"name": "t", "description": "d", "parameters": {}}]
    model._generate([])
    assert len(calls) == 1


def test_a_second_failure_still_answers_rather_than_raising(monkeypatch):
    """Two bad replies must not become a stack trace mid-turn."""
    from agent_runtime import chat_claude_cli
    from rag_pipeline import llm_claude_cli

    monkeypatch.setattr(llm_claude_cli, "call", lambda p, system=None: "still prose")
    monkeypatch.setattr(llm_claude_cli, "check_not_deployed", lambda: None)

    model = chat_claude_cli.build()
    model.tool_schemas = [{"name": "t", "description": "d", "parameters": {}}]
    result = model._generate([])
    assert result.generations[0].message.content == "still prose"
