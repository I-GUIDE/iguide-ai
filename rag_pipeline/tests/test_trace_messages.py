"""What the trace SAYS, checked against what the code does.

The audit that produced these: a message set accumulated over many changes, describing an
architecture that had moved. The worst of it named a component that does not exist — the
supervisor arm emitted "Orchestrator agent started" for a function that builds three callables
and calls run_supervisor. That string is TRUE on the legacy arm, where an orchestrator LLM
really is wrapped over the sub-agents as tools; it was copied across and never revisited.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_the_supervisor_arm_claims_no_orchestrator():
    from agent_runtime.supervisor import orchestration

    src = inspect.getsource(orchestration.run_supervisor_orchestration)
    # The MESSAGE values, not the source text: the comment above the emission quotes the old
    # string to say why it was wrong, and that is worth keeping.
    assert '"message": "Orchestrator agent' not in src, (
        "there is no orchestrator agent on this arm: no LLM, no decision, nothing an agent does")
    assert '"message": "Supervisor started"' in src
    assert '"message": "Supervisor finished"' in src


def test_the_pair_opens_and_closes_with_the_same_name():
    """It opened as the orchestrator and closed as "Supervisor graph completed" — two names for
    one bracket, which reads as two different things happening."""
    from agent_runtime.supervisor import orchestration

    src = inspect.getsource(orchestration.run_supervisor_orchestration)
    assert "Supervisor graph completed" not in src
    assert src.count('agent_role="supervisor"') == 2


def test_no_arm_claims_an_orchestrator_agent_any_more():
    """"Orchestrator agent started" was accurate on the agents-as-tools arm, where an
    orchestrator LLM really did call the other agents as tools. That arm is gone, so the string
    has no true home left — and a trace line naming a component that does not exist is how a
    reader builds the wrong mental model of the graph."""
    import agent_runtime

    # Scanned as STRING CONSTANTS via the AST, not as text: the comment in
    # supervisor/orchestration.py that records why this was renamed contains the phrase, and a
    # grep-shaped test would fail on the explanation for its own existence.
    import ast

    root = Path(agent_runtime.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "Orchestrator agent started":
                offenders.append(path.name)
    assert not offenders, offenders


def test_the_route_line_describes_the_request_not_the_graph():
    """"Routed to orchestrate" named a node in the triage graph. `fast` and `capabilities` were
    worse than jargon — each duplicated the destination node's own first line."""
    from agent_runtime import orchestrator_graph

    src = inspect.getsource(orchestrator_graph.build_orchestrator_graph)
    assert '"orchestrate": "Routed to the full agent"' in src
    assert 'f"Routed to {route}"' in src, "kept as the fallback for an unmapped route"


# --- the supervisor's reasons -------------------------------------------------------------
#
# `supervisor -> analyze (decision)`: an arrow, two node names, and a parenthetical whose
# commonest value means "no special reason". The other five values are the most informative
# thing in the trace — they say why the loop declined to do the obvious thing.

@pytest.mark.parametrize("nxt,why,expected", [
    ("analyze", "decision", None),
    ("done", "max_steps", "Stopping: this turn reached its step limit"),
    ("done", "search exhausted", "Stopping: the knowledge base has nothing further to give"),
    ("analyze", "nothing has run yet", "Starting with analyze: nothing has run yet this turn"),
    ("done", "no-progress repeat (analyze)",
     "Stopping: analyze would repeat with nothing new to work from"),
    ("code", "request by analysis_agent", "Running code, asked for by analysis_agent"),
])
def test_a_decision_reads_as_a_reason(nxt, why, expected):
    from agent_runtime.supervisor.graph import _decision_sentence

    assert _decision_sentence(nxt, why) == expected


def test_the_ordinary_decision_says_nothing_because_there_is_nothing_to_say():
    """None, not a sentence: "(decision)" means the decider simply chose, and a row per step
    saying so is the bookkeeping this whole pass removes."""
    from agent_runtime.supervisor.graph import _decision_sentence

    assert _decision_sentence("analyze", "decision") is None


def test_the_repeated_action_is_read_from_the_reason_not_from_next():
    """By the time this runs, `nxt` has been overwritten with "done" — so reading it would say
    "done would repeat", which is wrong about what the loop declined to do."""
    from agent_runtime.supervisor.graph import _decision_sentence

    assert "analyze would repeat" in _decision_sentence("done", "no-progress repeat (analyze)")
    assert "that step" in _decision_sentence("done", "no-progress repeat")


def test_an_unknown_reason_still_reaches_the_reader():
    """A reason added later must not vanish silently just because nobody wrote a sentence."""
    from agent_runtime.supervisor.graph import _decision_sentence

    assert _decision_sentence("done", "some new guard") == "done: some new guard"


def test_the_decision_event_is_forwarded_to_clients():
    """It is emitted as `decision`, which api/server.py has to allowlist or the sentence never
    leaves the server — the same way tool_dead_end was emitted for months and dropped here."""
    import pathlib

    src = pathlib.Path("api/server.py").read_text()
    block = src[src.index('if event_name in {\n                        "llm_interaction"'):]
    assert '"decision",' in block[:800]


def test_the_node_frame_is_documented():
    """It carries most of the progress text, and the streaming example never mentioned it — a
    client written from these docs drops that text into its default branch."""
    import pathlib

    src = pathlib.Path("api/server.py").read_text()
    assert "- `node`: graph node lifecycle" in src
    assert "event: node" in src


def test_no_ascii_ellipsis_is_left_in_a_status_string():
    import pathlib

    src = pathlib.Path("api/server.py").read_text()
    assert '"Updating memory..."' not in src
    assert '"status": "Updating memory"' in src
