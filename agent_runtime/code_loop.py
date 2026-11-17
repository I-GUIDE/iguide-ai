"""The code peer's own tool-calling loop (architecture stage 94).

Phase 2 of the agent's own framework. The code peer was a ``create_agent`` graph: a LangGraph
state machine whose model step, tool step and five middleware wrappers decided, between them,
when the run ends, what the model is sent, and what a tool failure looks like. Four rules the
run depends on lived in places the loop did not own:

* **the step limit** was LangGraph's ``recursion_limit`` (60 graph steps). Reaching it raised
  ``GraphRecursionError``, which ``invoke_agent_with_payload_fallback`` turned into
  ``AgentInvocationError`` and the code node into a peer FAILURE with ``answer: ""``: every
  result the run had recorded was discarded with it;
* **the dead-end rules** (a repeated call, two steps that add nothing) were middleware that
  could only return a message from inside the model step;
* **Stop** reached the loop through a callback that LangChain's callback manager swallows
  unless the exception is a ``BaseException`` (stage 78);
* **interrupt-and-correct** (the stage 78 review) lives inside ``execute_code``, which was the
  one part that did not need the graph at all.

This module is that loop, written out: one function that calls the model, runs its tool
calls, and stops for a reason it can name. ``AGENT_CODE_LOOP=own`` selects it (the default is
``langchain``, the graph). The swap is the two executor-construction lines in
``supervisor.graph.default_code_fn``, the same place ``AGENT_CODE_PEER`` swaps the whole peer
for a CLI; the supervisor graph, its nodes and edges and the checkpointer are not touched.

What stays LangChain, on purpose, so the comparison measures the loop and nothing else:

* **the tools** are the same ``StructuredTool`` objects from the same factories, and
  ``accept_null_defaults`` (tool_args) keeps working: it rewrites a function's signature before
  ``StructuredTool.from_function`` infers the schema, and this loop infers nothing. A call is
  run with ``tool.invoke(tool_call, config)``, which validates the arguments against that same
  schema, fires the trace callbacks (``tool_call``, ``tool_result``, ``map_layer``) and returns
  a ``ToolMessage``;
* **the model** is whatever ``build_default_llm`` / ``build_llm`` built (the own client of
  stage 88 or ``ChatOpenAI``), called through ``bind_tools(...).invoke(...)`` with the turn's
  callbacks attached, so the stage 59 cost handler prices each call unchanged;
* **the messages** are LangChain messages, so ``extract_search_artifacts`` and
  ``extract_final_answer`` read a run exactly as they read the graph's, and the result the
  code node returns keeps its flat ``answer`` / ``tool_calls`` / ``tool_results`` shape;
* **the turn log** (``turn_log``) is still the record: the repeat memo, the progress rule and
  the brief in the system message are read from it through the same functions the middleware
  uses (``run_tool_with_memo``, ``progress_stop_text``, ``_BRIEF_HEADER``).

What the loop owns:

* **an explicit step limit**, ``AGENT_CODE_LOOP_MAX_STEPS`` model calls per run (default 30,
  the graph's 60 steps halved: a model step and a tool step were two). At the limit the run
  ENDS WITH A MESSAGE that says so and keeps every recorded result;
* **the progress rule and the repeat memo**, applied by the loop itself before each model call
  and around each tool call;
* **Stop**, checked between steps here, mid-call by the own client, and inside a code run by
  the sandbox: ``TurnCancelled`` is a ``BaseException`` nothing in this module catches;
* **what the model is sent**: the system prompt plus the turn's brief, the thread's history
  repaired (``repair_tool_call_sequence``) and trimmed to the same budget the middleware
  derives (``_derive_budget`` / ``trim_to_budget``), without rewriting the stored history;
* **tool failures**: an unknown tool and a tool exception are answered with the texts
  LangGraph's tool node uses, so the model sees what it sees today; an unparseable tool call
  (``invalid_tool_calls``) is answered with its parse error instead of being sent back as a
  call with no result;
* **the thread's history**: a process-local, bounded store keyed by the same
  ``child_thread_id(thread_id, "code")`` the graph used, so a conversation's code thread
  still accumulates across turns and is evicted at the same cap.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime import turn_control, turn_log

logger = logging.getLogger(__name__)

LOOP_ENV = "AGENT_CODE_LOOP"                 # own | langchain (default)
MAX_STEPS_ENV = "AGENT_CODE_LOOP_MAX_STEPS"  # model calls per run
DEFAULT_MAX_STEPS = 30

# LangGraph's tool node's own words (langgraph.prebuilt.tool_node), so a failure reads the same
# to the model under either loop.
INVALID_TOOL_NAME_TEXT = "Error: {requested_tool} is not a valid tool, try one of [{available_tools}]."
TOOL_ERROR_TEXT = "Error: {error}\n Please fix your mistakes."

_LOGGED: List[bool] = []


def loop_choice() -> str:
    """``own`` or ``langchain``: which loop runs the code peer."""
    value = (os.getenv(LOOP_ENV) or "").strip().lower()
    return "own" if value == "own" else "langchain"


def selected() -> bool:
    return loop_choice() == "own"


def max_steps() -> int:
    raw = (os.getenv(MAX_STEPS_ENV) or "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MAX_STEPS
    except ValueError:
        logger.warning("%s=%r is not a number; using %d", MAX_STEPS_ENV, raw, DEFAULT_MAX_STEPS)
        return DEFAULT_MAX_STEPS


def step_limit_text(limit: int) -> str:
    """What a run ended by the step limit returns as its answer."""
    return (f"[Stopped: this run reached its limit of {limit} model calls without a final "
            "answer. The results recorded above are what it produced.]")


# --- the thread store -------------------------------------------------------------------------
class ThreadStore:
    """Each code thread's messages, process-local, least-recently-used threads evicted past a
    cap. The same contract as ``BoundedInMemorySaver`` (``AGENT_CHECKPOINT_MAX_THREADS``,
    default 500): multi-turn continuity needs sticky sessions across workers."""

    def __init__(self, max_threads: Optional[int] = None) -> None:
        if max_threads is None:
            try:
                max_threads = max(1, int(os.getenv("AGENT_CHECKPOINT_MAX_THREADS", "500")))
            except (TypeError, ValueError):
                max_threads = 500
        self.max_threads = max_threads
        self._threads: "OrderedDict[str, List[BaseMessage]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> List[BaseMessage]:
        with self._lock:
            messages = self._threads.get(key)
            if messages is not None:
                self._threads.move_to_end(key)
            return list(messages or [])

    def put(self, key: str, messages: Sequence[BaseMessage]) -> None:
        with self._lock:
            self._threads[key] = list(messages)
            self._threads.move_to_end(key)
            while len(self._threads) > self.max_threads:
                self._threads.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._threads.clear()


THREADS = ThreadStore()


def _with_id(message: BaseMessage) -> BaseMessage:
    if getattr(message, "id", None) is None:
        message.id = str(uuid.uuid4())
    return message


# --- the session --------------------------------------------------------------------------------
class OwnPeerSession:
    """One code thread for the duration of one turn, with ``PeerSession``'s face: ``run()``
    returns what THAT invocation produced, ``turn_artifacts`` accumulates the turn."""

    def __init__(self, llm: Any, tools: Sequence[Any], system_prompt: str,
                 thread_key: Optional[str]) -> None:
        self.llm = llm
        self.tools: List[Any] = list(tools)
        self.system_prompt = system_prompt
        self.thread_key = thread_key
        self._by_name: Dict[str, Any] = {str(getattr(t, "name", "")): t for t in self.tools}
        self._bound = llm.bind_tools(self.tools) if self.tools else llm
        self.turn_artifacts: Dict[str, List[Dict[str, Any]]] = {"tool_calls": [], "tool_results": []}
        self.steps = 0          # model calls made by this session's runs

    # -- the public face ------------------------------------------------------------------
    def run(self, query: str, *, chat_history: Optional[List[Any]] = None) -> Any:
        from agent_runtime.executor_factory import PeerRun
        from agent_runtime.runtime_utils import extract_final_answer, extract_search_artifacts

        history = THREADS.get(self.thread_key) if self.thread_key else []
        start = len(history)
        history.append(_with_id(HumanMessage(content=query)))
        try:
            self._loop(history)
        finally:
            # What ran is kept even when the run ends by Stop or a model failure: the next turn
            # of this thread continues from a repaired history, as the graph's did.
            if self.thread_key:
                THREADS.put(self.thread_key, history)
        scoped = {"messages": history[start:]}
        artifacts = extract_search_artifacts(scoped)
        for key in ("tool_calls", "tool_results"):
            self.turn_artifacts[key].extend(artifacts.get(key) or [])
        return PeerRun(resp={"messages": list(history)}, artifacts=artifacts,
                       answer=extract_final_answer(scoped) or "")

    # -- the loop -------------------------------------------------------------------------
    def _loop(self, history: List[BaseMessage]) -> None:
        from agent_runtime.executor_factory import progress_stop_text
        from agent_runtime.streaming_trace import attach_streaming_callbacks

        log, run = turn_log.active(), turn_log.active_run()
        limit = max_steps()
        for step in range(1, limit + 1):
            turn_control.raise_if_cancelled()
            if log is not None:
                try:
                    log.close_step(run)
                    if log.exhausted(run):
                        logger.warning("code loop run %s ended: %d consecutive steps added "
                                       "nothing", run, turn_log.PROGRESS_LIMIT)
                        history.append(_with_id(AIMessage(content=progress_stop_text())))
                        return
                except Exception:  # noqa: BLE001 - progress accounting must never cost the call
                    pass
            system = self._system_message()
            messages = self._request_messages(system, history)
            config = attach_streaming_callbacks({})
            self.steps += 1
            logger.info("code loop: step %d/%d, %d message(s), %d tool(s) bound",
                        step, limit, len(messages), len(self.tools))
            reply = self._bound.invoke([system, *messages], config=config)
            ai = reply if isinstance(reply, AIMessage) else AIMessage(content=str(reply))
            history.append(_with_id(ai))
            calls = list(getattr(ai, "tool_calls", None) or [])
            invalid = list(getattr(ai, "invalid_tool_calls", None) or [])
            if not calls and not invalid:
                return
            for bad in invalid:
                history.append(_with_id(ToolMessage(
                    content=TOOL_ERROR_TEXT.format(error=bad.get("error") or "invalid tool call"),
                    tool_call_id=bad.get("id") or str(uuid.uuid4()),
                    name=bad.get("name") or "", status="error")))
            for call in calls:
                history.append(_with_id(self._run_call(call, config)))
        logger.warning("code loop run %s ended at its step limit (%d model calls)", run, limit)
        history.append(_with_id(AIMessage(content=step_limit_text(limit))))

    def _system_message(self) -> SystemMessage:
        from agent_runtime.executor_factory import _BRIEF_HEADER

        brief = turn_log.active_brief()
        content = self.system_prompt + (_BRIEF_HEADER + brief if brief else "")
        return SystemMessage(content=content)

    def _request_messages(self, system: SystemMessage,
                          history: List[BaseMessage]) -> List[BaseMessage]:
        """The history as sent: repaired and under the budget. The stored history is untouched."""
        from agent_runtime import executor_factory as ef
        from agent_runtime.runtime_utils import repair_tool_call_sequence

        messages, _ = repair_tool_call_sequence(list(history))
        messages = list(messages)
        if ef._context_budget() < 0:              # explicitly disabled
            return messages
        try:
            request = SimpleNamespace(model=self.llm, system_message=system, tools=self.tools,
                                      messages=messages)
            trimmed = ef.trim_to_budget(messages, ef._derive_budget(request))
        except Exception:  # noqa: BLE001 - counting must never break a call
            logger.debug("code loop: budget trim skipped", exc_info=True)
            return messages
        return trimmed if trimmed is not None else messages

    def _run_call(self, call: Dict[str, Any], config: Dict[str, Any]) -> ToolMessage:
        from agent_runtime import lineage
        from agent_runtime.executor_factory import run_tool_with_memo

        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        call_id = call.get("id") or str(uuid.uuid4())
        turn_control.raise_if_cancelled()
        tool = self._by_name.get(name)
        if tool is None:
            return ToolMessage(
                content=INVALID_TOOL_NAME_TEXT.format(
                    requested_tool=name, available_tools=", ".join(self._by_name)),
                tool_call_id=call_id, name=name, status="error")

        def invoke() -> Any:
            return tool.invoke({"name": name, "args": args, "id": call_id, "type": "tool_call"},
                               config=config)

        with lineage.calling(call_id, name, args, description=getattr(tool, "description", None)):
            try:
                result = run_tool_with_memo(name, args, call_id, invoke)
            except Exception as exc:  # noqa: BLE001 - the model gets the error, as under the graph
                return ToolMessage(content=TOOL_ERROR_TEXT.format(error=repr(exc)),
                                   tool_call_id=call_id, name=name, status="error")
        if isinstance(result, ToolMessage):
            return result
        return ToolMessage(content=result if isinstance(result, str) else str(result),
                           tool_call_id=call_id, name=name)


def open_session(*, llm: Optional[Any], tools: Sequence[Any], system_prompt_override: str,
                 thread_key: Optional[str], skill_roots: Optional[List[str]] = None,
                 prepare: Optional[Callable[..., List[Any]]] = None) -> OwnPeerSession:
    """A session over the same prepared tools and prompt ``build_agent_executor`` would bind."""
    from agent_runtime.executor_factory import (
        build_default_llm, peer_system_prompt, prepare_peer_tools,
    )

    if not _LOGGED:
        _LOGGED.append(True)
        logger.info("code loop: own (AGENT_CODE_LOOP=own, code_loop.py)")
    prepared = (prepare or prepare_peer_tools)(list(tools), skill_roots=skill_roots)
    return OwnPeerSession(llm or build_default_llm(), prepared,
                          peer_system_prompt(system_prompt_override), thread_key)


__all__ = ["DEFAULT_MAX_STEPS", "LOOP_ENV", "MAX_STEPS_ENV", "OwnPeerSession", "THREADS",
           "ThreadStore", "loop_choice", "max_steps", "open_session", "selected",
           "step_limit_text"]
