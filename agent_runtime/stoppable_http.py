"""An httpx client whose model requests end the moment their turn is stopped (Stage 95.1).

Stage 95 made Stop take effect at the supervisor's next step and refused every model request
sent after it, the openai SDK's own retries included. What it left was the request already in
flight: measured live, a code-peer call stalled at the provider ran on to its 45 s timeout after
Stop, so the turn ended 42.5 s after the person pressed it (with the default 240 s timeout it
would have been up to four minutes), and the provider kept the connection, and possibly the
generation, open the whole time.

The openai SDK and ``ChatOpenAI`` give no way to abort a synchronous call from outside, so the
abort lives one level down, in the transport. For a request made inside a turn,
``StoppableTransport`` runs the send, and every later read of the response body, on a
worker thread, and the calling thread waits on whichever comes first: the worker, or the turn's
Stop. On Stop it shuts the request's socket down, which is what wakes a ``recv`` blocked in the
worker (``close()`` from another thread does not, on Linux), and raises ``TurnCancelled``.

Every request has a *lane* of its own while it is in flight: one ``httpx.HTTPTransport`` with
its own connection pool, lent to one request at a time and given back when the response is
closed, so idle lanes keep their keep-alive connections and an abort shuts exactly the sockets
of the request it stops. The same shape as the Phase 2 model client's ``_Lease``.

The classes are built against the HTTP library the openai SDK itself uses (:func:`build`):
``httpx``/``httpcore`` for openai 2.x, ``httpx2``/``httpcore2`` for the 3.x the images pin. A
transport from the other library fails an ``isinstance`` assertion on every request.

A request made outside any turn (a test, a background job) is sent on the calling thread as
before, with no worker. ``TurnCancelled`` is a ``BaseException``, so the SDK's retry loop, which
catches ``Exception``, does not swallow it.
"""

from __future__ import annotations

import importlib
import logging
import queue
import socket
import threading
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional

from agent_runtime import turn_control

logger = logging.getLogger(__name__)

# How often a waiting caller looks at the turn's Stop. Stop is noticed within this.
POLL_S = 0.05
# Idle lanes kept for reuse; any beyond this are closed when given back.
_MAX_IDLE_LANES = 16


def _cancelled(handle: turn_control.TurnHandle) -> turn_control.TurnCancelled:
    return turn_control.TurnCancelled(handle.reason or "stopped by the user")


def _wait(handle: turn_control.TurnHandle, done: threading.Event, lane: Any) -> None:
    """Block until ``done`` or the turn's Stop; on Stop abort the lane and raise."""
    while not done.wait(POLL_S):
        if handle.cancelled:
            lane.abort()
            raise _cancelled(handle)


_BUILT: Dict[str, SimpleNamespace] = {}
_BUILT_LOCK = threading.Lock()


def build(httpx_name: str = "httpx") -> SimpleNamespace:
    """The transport classes for one HTTP library (``httpx`` or ``httpx2``), built once."""
    with _BUILT_LOCK:
        if httpx_name not in _BUILT:
            hx = importlib.import_module(httpx_name)
            # The httpcore that library drives, read off its own pool rather than assumed.
            probe = hx.HTTPTransport()
            hc = importlib.import_module(type(probe._pool).__module__.split(".")[0])
            probe.close()
            _BUILT[httpx_name] = _define(hx, hc)
        return _BUILT[httpx_name]


def _define(hx: Any, hc: Any) -> SimpleNamespace:
    class _TrackingBackend(hc.SyncBackend):
        """httpcore's own backend, remembering the sockets it opens so an abort can shut them."""

        def __init__(self) -> None:
            super().__init__()
            self._socks: List[socket.socket] = []
            self._lock = threading.Lock()

        def connect_tcp(self, *args: Any, **kwargs: Any) -> Any:
            stream = super().connect_tcp(*args, **kwargs)
            sock = stream.get_extra_info("socket")
            if sock is not None:
                with self._lock:
                    self._socks = [s for s in self._socks if s.fileno() != -1] + [sock]
            return stream

        def abort(self) -> None:
            with self._lock:
                socks, self._socks = self._socks, []
            for sock in socks:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


    class _Lane:
        """One HTTP transport with its own pool, lent to one request at a time."""

        def __init__(self, make_transport: Callable[[], Any]) -> None:
            self.backend = _TrackingBackend()
            self.transport = make_transport()
            # httpx 0.28 builds its httpcore pool in __init__ with the default backend and takes no
            # argument for another. Without the swap an abort still releases the caller at once,
            # but the stopped request's worker blocks on until its own timeout.
            # test_stop_interrupts_in_flight pins it, so an httpx that renames it fails loudly.
            pool = getattr(self.transport, "_pool", None)
            self.abortable = pool is not None and hasattr(pool, "_network_backend")
            if self.abortable:
                pool._network_backend = self.backend

        def abort(self) -> None:
            self.backend.abort()
            try:
                self.transport.close()
            except Exception:  # noqa: BLE001 - closing a transport being discarded
                pass

        def close(self) -> None:
            try:
                self.transport.close()
            except Exception:  # noqa: BLE001
                pass


    class _StoppableStream(hx.SyncByteStream):
        """A response body whose reads give way to Stop, and whose close returns the lane."""

        def __init__(self, inner: Any, lane: _Lane, owner: Any,
                     handle: Optional[turn_control.TurnHandle]) -> None:
            self._inner = inner
            self._lane = lane
            self._owner = owner
            self._handle = handle
            self._clean = False
            self._released = False

        def __iter__(self) -> Iterator[bytes]:
            if self._handle is None:
                yield from self._inner
                self._clean = True
                return
            handle = self._handle
            chunks: "queue.Queue[tuple]" = queue.Queue()

            def pump() -> None:
                try:
                    for chunk in self._inner:
                        chunks.put(("chunk", chunk))
                    chunks.put(("end", None))
                except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
                    chunks.put(("error", exc))

            if handle.cancelled:
                raise _cancelled(handle)
            threading.Thread(target=pump, name="model-response-read", daemon=True).start()
            while True:
                try:
                    kind, value = chunks.get(timeout=POLL_S)
                except queue.Empty:
                    if handle.cancelled:
                        raise _cancelled(handle)        # close() below aborts the lane
                    continue
                if kind == "chunk":
                    yield value
                elif kind == "end":
                    self._clean = True
                    return
                else:
                    raise value

        def close(self) -> None:
            if self._released:
                return
            self._released = True
            if self._clean:
                try:
                    self._inner.close()
                except Exception:  # noqa: BLE001
                    self._lane.abort()
                    return
                self._owner._give_back(self._lane)
            else:
                # Not read to the end: the connection is mid-response and cannot be reused.
                self._lane.abort()


    class StoppableTransport(hx.BaseTransport):
        """Sends each request on a lane of its own, and lets the turn's Stop abort it in flight."""

        def __init__(self, make_transport: Optional[Callable[[], Any]] = None) -> None:
            self._make = make_transport or (lambda: hx.HTTPTransport())
            self._idle: List[_Lane] = []
            self._lock = threading.Lock()

        def _take(self) -> _Lane:
            with self._lock:
                if self._idle:
                    return self._idle.pop()
            return _Lane(self._make)

        def _give_back(self, lane: _Lane) -> None:
            with self._lock:
                if len(self._idle) < _MAX_IDLE_LANES:
                    self._idle.append(lane)
                    return
            lane.close()

        def handle_request(self, request: Any) -> Any:
            handle = turn_control.current()
            if handle is not None and handle.cancelled:
                # Also covers the openai SDK's retry of a request that failed after Stop.
                raise _cancelled(handle)
            lane = self._take()
            if handle is None:
                try:
                    response = lane.transport.handle_request(request)
                except BaseException:
                    lane.abort()
                    raise
                return self._wrap(response, lane, None)

            box: dict = {}
            done = threading.Event()

            def send() -> None:
                try:
                    box["response"] = lane.transport.handle_request(request)
                except BaseException as exc:  # noqa: BLE001 - handed to the waiting thread
                    box["error"] = exc
                finally:
                    done.set()

            threading.Thread(target=send, name="model-request", daemon=True).start()
            _wait(handle, done, lane)
            if "error" in box:
                lane.abort()
                raise box["error"]
            return self._wrap(box["response"], lane, handle)

        def _wrap(self, response: Any, lane: _Lane,
                  handle: Optional[turn_control.TurnHandle]) -> Any:
            return hx.Response(
                status_code=response.status_code, headers=response.headers,
                stream=_StoppableStream(response.stream, lane, self, handle),  # type: ignore[arg-type]
                extensions=response.extensions)

        def close(self) -> None:
            with self._lock:
                idle, self._idle = self._idle, []
            for lane in idle:
                lane.close()

    return SimpleNamespace(StoppableTransport=StoppableTransport, Lane=_Lane, httpx=hx)


_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()


def stoppable_http_client() -> Any:
    """The shared httpx client every ``ChatOpenAI`` built here sends through.

    The openai SDK's own default client (its timeout, connection limits and redirects), with the
    transport swapped for ``StoppableTransport`` from the same HTTP library. One instance per process, as LangChain
    shares its default.
    """
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            import openai

            library = openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0]
            _CLIENT = openai.DefaultHttpxClient(transport=build(library).StoppableTransport())
        return _CLIENT


__all__ = ["POLL_S", "build", "stoppable_http_client"]
