"""The supervisor's description of its peers must not drift behind the tools they actually bind.

The decider chooses a capability from a hand-written paragraph describing what each peer can do.
Nothing derives that paragraph from the peers' real toolsets, so it drifts — and silently. When
four terrain tools were added and bound to BOTH peers, the paragraph was never updated, so the
supervisor did not know its analysis peer could compute a DEM. Asked for one, it searched the
knowledge base for "digital elevation model" instead. That was not a weak model guessing badly;
it was a correct decision from a stale description.

The inventory is now GENERATED from ``agent_runtime.capability_registry``, so the prompt cannot
fall behind the registry. What can still fall behind is the registry itself, and that is what
the first tests hold: the factories the registry claims must be exactly the factories the peer
builders call. Bind a new toolset without describing it and this fails with its name.

Everything after that reads the PROMPT THE DECIDER IS SENT, captured from ``default_decide_fn``
itself. An earlier version of this file composed its own view instead — the hand-written framing
plus ``describe("analyze")`` plus ``describe("code")`` — and checked that. But the real decider
never called ``describe("code")``: its ``code`` line was hand-written. So a code-only registry
entry passed here and never reached the decider, and the test certified an artifact production
did not use. The hand-written line was also wrong on its own terms: it promised "saved
workflows" (the code peer binds no tool that runs one), skills the deployed image did not ship,
and the LangChain toolkit to CLI code peers that have none of it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRAPH = Path(__file__).resolve().parents[2] / "agent_runtime" / "supervisor" / "graph.py"
SOURCE = GRAPH.read_text(encoding="utf-8")

# What the supervisor must SAY for each toolset its peers bind. The registry clause for the
# toolset has to contain one of the listed terms — they are the words a model would need to see
# to connect a request to that capability — and that clause has to reach the decider.
#
# Adding a toolset means adding a line here AND words to the registry. That is the whole point:
# the cost of binding a capability the supervisor cannot describe is one failing test, paid
# immediately, instead of a class of misrouted turns discovered months later.
REQUIRED_TERMS = {
    "make_terrain_tools": ("dem", "elevation", "terrain", "slope"),
    "make_overlay_tools": ("overlay", "buffer", "clip", "dissolve"),
    "make_aggregate_tools": ("aggregat", "summaris", "summariz"),
    "make_temporal_tools": ("temporal", "time series", "over time"),
    "make_spatial_stats_tools": ("statistic", "autocorrelat", "cluster"),
    "make_geo_analysis_tools": ("gis", "geospatial", "spatial"),
    "make_admin_boundary_tools": ("boundary", "boundaries", "administrative"),
    "make_rs_embed_tools": ("embedding", "embed"),
    "make_rs_embed_zonal_tools": ("zone", "zonal", "segment"),
    "make_langchain_qgis_tools": ("qgis", "pyqgis"),
    "make_langchain_geocode_tools": ("geocod", "place name", "address"),
    "make_langchain_geo_tools": ("reproject", "vector", "geojson"),
    "make_langchain_granular_tools": ("search", "retriev"),
    "make_langchain_file_tools": ("file", "upload"),
    "make_conversation_file_tools": ("file", "upload"),
    "make_code_execution_tools": ("code", "execut"),
    "make_skill_tools": ("skill",),
    "make_langchain_mcp_tools": ("mcp", "external tool", "qgis"),
}

ALL_ACTIONS = {"available_actions": ["search", "analyze", "code", "done"]}


@pytest.fixture(autouse=True)
def _no_inherited_peer_or_skill_settings(monkeypatch):
    """The decider's code line depends on AGENT_CODE_PEER and on skill discovery; a developer
    shell that sets either must not decide what these tests see."""
    for var in ("AGENT_CODE_PEER", "AGENT_SKILLS_ENABLED", "AGENT_SKILL_PATHS",
                "AGENT_SKILLS_PATHS", "AGENT_CLAUDE_NETWORK", "AGENT_OPENCODE_NETWORK"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def one_skill(tmp_path) -> list:
    """A skill root holding exactly one valid skill, so the skills clause is deterministic
    rather than a function of whatever the checkout happens to ship."""
    skill = tmp_path / "skills" / "demo-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Use when checking that skills reach the decider.\n"
        "---\n\n# Demo skill\n", encoding="utf-8")
    return [str(tmp_path / "skills")]


@pytest.fixture
def no_skills(tmp_path) -> list:
    """A skill root with nothing in it: what the deployed image amounted to."""
    empty = tmp_path / "empty-skills"
    empty.mkdir()
    return [str(empty)]


class _Recorder:
    """An LLM that keeps the prompt it was sent and answers `done`."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return '{"next": "done", "reason": "recorded"}'


def _decider_prompt(*, code_peer=None, skill_roots=None) -> str:
    """What the decider is ACTUALLY sent: built by ``default_decide_fn`` itself, the way
    ``decide()`` builds it for a live request. Composing the description here instead is the
    mistake this file used to make."""
    from agent_runtime.supervisor.graph import default_decide_fn

    llm = _Recorder()
    assert default_decide_fn(llm=llm, code_peer=code_peer, skill_roots=skill_roots)(
        {"query": "q"}, dict(ALL_ACTIONS)) == "done"
    assert len(llm.prompts) == 1
    return llm.prompts[0]


def _line(prompt: str, action: str) -> str:
    """The one line of the decider's action menu that describes *action*."""
    lines = [ln for ln in prompt.splitlines() if ln.startswith(f"- {action}:")]
    assert len(lines) == 1, f"expected one '- {action}:' line in the decider prompt, got {lines}"
    return lines[0]


def _bound_toolsets() -> set:
    """Every `make_*_tools(` factory the supervisor graph actually calls."""
    return set(re.findall(r"\b(make_[a-z0-9_]+_tools)\s*\(", SOURCE))


def _registry_factories() -> set:
    from agent_runtime.capability_registry import factories
    return factories()


def _clauses_for(factory: str) -> list:
    from agent_runtime.capability_registry import CAPABILITIES
    return [t.summary for caps in CAPABILITIES.values() for t in caps if t.factory == factory]


# --- the registry against the peers' real bindings -------------------------------------------

def test_the_registry_describes_everything_the_peers_bind():
    """The drift that mattered: bound to a peer, invisible to the supervisor."""
    undescribed = _bound_toolsets() - _registry_factories()
    assert not undescribed, (
        f"these toolsets are bound to a peer but missing from capability_registry: "
        f"{sorted(undescribed)}. The supervisor cannot route work to a capability it has not "
        "been told about — that is how a DEM request became a knowledge-base search.")


def test_the_registry_does_not_claim_tools_the_peers_do_not_bind():
    """Drift in the other direction: promising a capability that is not there sends the
    supervisor to a peer that cannot deliver, and the turn fails further downstream where the
    cause is much harder to see."""
    phantom = _registry_factories() - _bound_toolsets()
    assert not phantom, (
        f"capability_registry claims these, but no peer binds them: {sorted(phantom)}")


def test_every_bound_toolset_has_a_stated_expectation():
    """A toolset nobody listed here is a toolset nobody thought about describing."""
    unlisted = _bound_toolsets() - set(REQUIRED_TERMS)
    assert not unlisted, (
        f"these toolsets are bound to a peer but absent from REQUIRED_TERMS: {sorted(unlisted)}. "
        "Add each one here with the words the supervisor should use for it, and add those words "
        "to the registry — otherwise the supervisor cannot route work to it.")


# --- the registry against the prompt the decider is actually sent ----------------------------

@pytest.mark.parametrize("capability", ["analyze", "code"])
def test_each_capability_line_carries_its_whole_inventory(capability, one_skill):
    """Every clause the registry produces for a capability is in THAT capability's line of the
    real prompt. This is the check the composed view could not make: it passed while the code
    line in production was a hand-written sentence that read none of them."""
    from agent_runtime.capability_registry import describe

    line = _line(_decider_prompt(skill_roots=one_skill), capability)
    inventory = describe(capability, skill_roots=one_skill)
    missing = [clause for clause in inventory.split("; ") if clause not in line]
    assert not missing, f"registry clauses for {capability} never reach the decider: {missing}"
    assert inventory in line


def test_a_code_only_entry_reaches_the_decider(monkeypatch):
    """The defect, reproduced. A toolset only the code peer binds is described ONLY by
    describe("code") — and the old decider never called it, so the entry existed for the test
    and not for the model."""
    import agent_runtime.capability_registry as reg

    marker = reg.Toolset("make_only_the_code_peer_has_this_tools",
                         "a capability only the code peer has")
    monkeypatch.setitem(reg.CAPABILITIES, "code", reg.CAPABILITIES["code"] + (marker,))
    prompt = _decider_prompt()
    assert marker.summary in _line(prompt, "code")
    assert marker.summary not in _line(prompt, "analyze"), "a code-only entry is not analyze's"


@pytest.mark.parametrize("factory", sorted(REQUIRED_TERMS))
def test_the_supervisor_can_describe_what_its_peers_bind(factory, one_skill):
    """The words, in the toolset's OWN clause, and that clause in the prompt. Checking for the
    words anywhere in the prompt let framing prose satisfy them: "run a workflow" in the analyze
    line answered for a skills clause that was never there."""
    if factory not in _bound_toolsets():
        pytest.skip(f"{factory} is not bound in this build")
    terms = REQUIRED_TERMS[factory]
    clauses = _clauses_for(factory)
    assert clauses, f"{factory} has no clause in capability_registry"
    for clause in clauses:
        assert any(term in clause.lower() for term in terms), (
            f"{factory}'s registry clause {clause!r} never says any of {terms}. The supervisor "
            "cannot route a request to a capability it has not been told about — this is "
            "exactly how a DEM request became a knowledge-base search.")
    prompt = _decider_prompt(skill_roots=one_skill)
    unseen = [clause for clause in clauses if clause not in prompt]
    assert not unseen, f"{factory} is described in the registry but not to the decider: {unseen}"


def test_the_inventory_actually_reaches_the_prompt():
    """The generator is not decorative: if it silently returned the fallback, every keyword
    test above would still pass against leftover hand-written prose."""
    from agent_runtime.capability_registry import describe
    from agent_runtime.supervisor.graph import _capability_inventory
    assert _capability_inventory("analyze") == describe("analyze")
    assert "elevation and terrain" in _capability_inventory("analyze")
    assert "elevation and terrain" in _line(_decider_prompt(), "code"), (
        "the code line is generated too, so the shared toolkit is in it")


def test_the_regression_that_prompted_this_guard():
    """Terrain shipped bound-but-undescribed. Named explicitly so it cannot quietly return."""
    assert "make_terrain_tools" in _bound_toolsets()
    line = _line(_decider_prompt(), "analyze").lower()
    assert any(t in line for t in ("dem", "elevation", "terrain")), (
        "the terrain toolset is bound but the supervisor's description of `analyze` does not "
        "mention elevation work")


# --- skills: described only when the peer would have some ------------------------------------

def test_skills_are_described_only_when_discovery_finds_one(one_skill, no_skills):
    """make_skill_tools binds nothing for an empty registry, and the deployed image shipped no
    skill roots at all, so the clause has to follow the same discovery over the same roots."""
    with_skill = _line(_decider_prompt(skill_roots=one_skill), "code")
    without = _line(_decider_prompt(skill_roots=no_skills), "code")
    assert "packaged skills" in with_skill
    assert "skill" not in without.lower(), without


def test_switching_skills_off_removes_the_clause(monkeypatch, one_skill):
    """AGENT_SKILLS_ENABLED=0 empties discovery, so it empties the description too."""
    monkeypatch.setenv("AGENT_SKILLS_ENABLED", "0")
    assert "skill" not in _line(_decider_prompt(skill_roots=one_skill), "code").lower()


def test_nothing_promises_saved_workflows(one_skill, no_skills):
    """The code peer cannot run a saved workflow. A notebook workflow packaged as a skill runs
    through an MCP tool, and the code peer never binds MCP tools. The phrase was the skill
    loader's registry clause, and the decider repeated it as a capability."""
    from agent_runtime.capability_registry import CAPABILITIES

    assert not any("saved workflow" in t.summary.lower()
                   for caps in CAPABILITIES.values() for t in caps)
    for roots in (one_skill, no_skills):
        for peer in (None, "claude", "opencode"):
            assert "saved workflow" not in _decider_prompt(
                code_peer=peer, skill_roots=roots).lower()


# --- the selected code peer is the one described ---------------------------------------------

@pytest.mark.parametrize("peer,name", [("claude", "Claude Code"), ("opencode", "opencode")])
def test_a_cli_code_peer_is_described_as_itself(peer, name, one_skill):
    """A CLI peer runs in its own container with network and binds none of the LangChain
    toolkit, so promising it that toolkit sends work to tools that are not there."""
    from agent_runtime.capability_registry import CAPABILITIES, CLI_PEER

    prompt = _decider_prompt(code_peer=peer, skill_roots=one_skill)
    line = _line(prompt, "code")
    assert f"({name})" in line
    assert "network access" in line
    assert all(clause in line for clause in CLI_PEER)
    claimed = [t.summary for t in CAPABILITIES["code"] if t.summary in line]
    assert not claimed, f"a CLI peer is described with toolsets it does not bind: {claimed}"
    assert "same toolkit as analyze" not in line
    assert "none of the tools listed for analyze" in line
    assert "cannot request another capability" in line
    # And the generic sentence about requests no longer offers the code peer as its example.
    assert "(e.g. code needs evidence)" not in prompt
    assert "(e.g. analyze needs evidence)" in prompt


def test_the_langchain_peer_keeps_the_request_example(one_skill):
    """The built-in peer DOES have request_capability, so its example is unchanged."""
    assert "(e.g. code needs evidence)" in _decider_prompt(skill_roots=one_skill)


def test_the_env_default_selects_the_description_when_the_request_names_none(monkeypatch):
    monkeypatch.setenv("AGENT_CODE_PEER", "opencode")
    assert "(opencode)" in _line(_decider_prompt(), "code")


def test_the_request_wins_over_the_env_default(monkeypatch):
    """Same precedence as the code node: a request naming langchain opts out of a CLI default,
    and a request naming a CLI overrides a different one."""
    monkeypatch.setenv("AGENT_CODE_PEER", "claude")
    line = _line(_decider_prompt(code_peer="langchain"), "code")
    assert "It can currently do: geocoding" in line and "Claude Code" not in line
    assert "(opencode)" in _line(_decider_prompt(code_peer="opencode"), "code")


def test_one_resolution_decides_what_runs_and_what_is_described(monkeypatch):
    """The decider and the code node must not each resolve the peer: two copies of
    `code_peer or AGENT_CODE_PEER` would let the supervisor describe one peer while another
    runs. Both go through _code_peer_backend, so replacing it moves both at once."""
    import agent_runtime.opencode_peer as ocp
    import agent_runtime.supervisor.graph as g

    asked = []
    monkeypatch.setattr(g, "_code_peer_backend", lambda code_peer=None: asked.append(code_peer)
                        or "opencode")
    ran = []
    monkeypatch.setattr(ocp, "run_opencode_code_peer",
                        lambda q, **kw: ran.append(q) or {"answer": "", "tool_calls": [],
                                                          "tool_results": []})
    assert "(opencode)" in _line(_decider_prompt(code_peer="from-the-request"), "code")
    g.default_code_fn(code_peer="from-the-request")("q", [], {})
    assert ran == ["q"], "the code node ran what the shared resolution chose"
    assert asked and set(asked) == {"from-the-request"}


@pytest.mark.parametrize("request_peer,env,expected", [
    (None, None, "langchain"),
    (None, "claude", "claude"),
    (None, "opencode", "opencode"),
    ("claude", "opencode", "claude"),
    ("langchain", "claude", "langchain"),
    ("Claude-Code", None, "claude"),
    ("something-else", None, "langchain"),
])
def test_the_resolution_itself(monkeypatch, request_peer, env, expected):
    from agent_runtime.supervisor.graph import _code_peer_backend

    if env is None:
        monkeypatch.delenv("AGENT_CODE_PEER", raising=False)
    else:
        monkeypatch.setenv("AGENT_CODE_PEER", env)
    assert _code_peer_backend(request_peer) == expected


def test_the_request_reaches_the_decider_through_orchestration(monkeypatch):
    """Per-request selection only helps if the request's code_peer reaches the decider: built
    by build_supervisor_graph's default, the decider would describe the env default while
    default_code_fn(code_peer=...) ran something else."""
    import agent_runtime.supervisor.graph as sg
    from agent_runtime.strategy import OrchestrationConfig
    from agent_runtime.supervisor.orchestration import run_supervisor_orchestration

    captured = {}

    def fake_run_supervisor(query, **kwargs):
        captured.update(kwargs)
        return {"final_answer": "", "audit": {}}

    monkeypatch.setattr(sg, "run_supervisor", fake_run_supervisor)
    for name in ("default_search_fn", "default_analyze_fn", "default_code_fn"):
        monkeypatch.setattr(sg, name, lambda **kwargs: (lambda *a, **k: None))
    llm = _Recorder()
    run_supervisor_orchestration("q", [], OrchestrationConfig(llm=llm, code_peer="claude"))
    captured["decide_fn"]({"query": "q"}, dict(ALL_ACTIONS))
    assert "(Claude Code)" in _line(llm.prompts[-1], "code")


# --- the CLI description held to the CLI's real binding --------------------------------------

@pytest.mark.parametrize("peer", ["claude", "opencode"])
def test_a_cli_peer_container_really_keeps_network(peer, tmp_path):
    """The decider is told a CLI peer has network access. That is true because neither
    build_docker_argv passes `--network none` — the reverse of execute_code. If a deployment
    hardening ever adds it, this fails here instead of the decider routing a fetch to a peer
    that can no longer make one."""
    from agent_runtime.capability_registry import CLI_PEER

    if peer == "claude":
        from agent_runtime.claude_peer import build_docker_argv
    else:
        from agent_runtime.opencode_peer import build_docker_argv
    argv = build_docker_argv(tmp_path, "probe", "model", "prompt")
    pairs = list(zip(argv, argv[1:]))
    assert ("--network", "none") not in pairs and "--network=none" not in argv
    assert any("network access" in clause for clause in CLI_PEER)
