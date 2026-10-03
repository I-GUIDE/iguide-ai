"""Reproduce the macOS crash that kills every child a long-lived agent process starts, and show
that ``agent_runtime.fork_safe`` does not hit it.

    PYTHONPATH="$PWD" python3 scripts/repro_macos_fork_crash.py          # a short report
    PYTHONPATH="$PWD" python3 scripts/repro_macos_fork_crash.py --json   # what the test reads

It does what one server turn did, in the same order:

1. holds a turn's worth of small Python objects (a parsed response, a frame of tool results);
2. reprojects once through geopandas in a worker thread, as ``add_map_layer`` does;
3. releases the objects, so CPython returns their emptied arenas and leaves free address space
   below the logging-preferences mapping that the reprojection's SQLite logging brought in;
4. from another worker thread starts ``pwd`` in a working directory twice: through
   ``fork_safe.run``, then through ``subprocess.run(cwd=...)``, which is what
   ``qgis_headless_tools._run_subprocess`` did when ``qgis_process`` returned -11.

On an affected Mac the second child faults inside fork, before exec. Such a child would leave a
crash report in ~/Library/Logs/DiagnosticReports, so SIGSEGV and SIGBUS are pointed at ``_exit``
first and it exits with status 11 (or 10) instead; ``--crash-report`` keeps the default action
and produces a real report to compare with the ones a server writes. The disposition is reset
by exec, so a child that gets that far runs with the default.

Without step 1 a short script survives: the child maps the preferences back at the address the
parent had them, and the stale pointer it reads happens to be valid again. That is why a
three-line pyproj-then-fork script never reproduced this. See ``agent_runtime/fork_safe.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Dict

from agent_runtime import fork_safe


def _exit_on_fault() -> None:
    """Turn a fault into ``_exit(signal number)``: same diagnosis, no crash report."""
    libc = ctypes.CDLL(None)
    libc.signal.restype = ctypes.c_void_p
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    exit_address = ctypes.cast(libc._exit, ctypes.c_void_p).value
    for sig in (signal.SIGSEGV, signal.SIGBUS):
        libc.signal(sig, exit_address)


def _in_worker_thread(fn: Callable[[], Any]) -> Any:
    box: Dict[str, Any] = {}

    def body() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - reported, not raised, so the other launch runs
            box["value"] = exc

    thread = threading.Thread(target=body)
    thread.start()
    thread.join()
    return box["value"]


def _reproject() -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    gpd.GeoDataFrame(geometry=[Point(-88.2, 40.1)], crs="EPSG:4326").to_crs("EPSG:26916")


def _outcome(result: Any) -> Dict[str, Any]:
    if isinstance(result, Exception):
        return {"returncode": None, "stdout": "", "error": f"{type(result).__name__}: {result}"}
    return {"returncode": result.returncode, "stdout": (result.stdout or "").strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--objects", type=int, default=500_000,
                        help="small objects alive across the reprojection (0 = a short script)")
    parser.add_argument("--json", action="store_true", help="print one JSON line")
    parser.add_argument("--crash-report", action="store_true",
                        help="let a faulting child die normally, so macOS writes a crash report")
    args = parser.parse_args()

    if not args.crash_report:
        _exit_on_fault()
    turn_data = [{"row": i} for i in range(args.objects)]
    _in_worker_thread(_reproject)
    del turn_data
    gc.collect()

    cwd = os.path.realpath(tempfile.mkdtemp(prefix="fork_crash_"))
    guarded = _in_worker_thread(lambda: fork_safe.run(["pwd"], cwd=cwd, capture_output=True,
                                                      text=True, timeout=60))
    plain = _in_worker_thread(lambda: subprocess.run(["pwd"], cwd=cwd, capture_output=True,
                                                     text=True, timeout=60))
    os.rmdir(cwd)

    import pyproj

    report = {"platform": sys.platform, "objects": args.objects, "cwd": cwd,
              "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str,
              "fork_safe": _outcome(guarded), "subprocess": _outcome(plain)}
    if args.json:
        print(json.dumps(report))
    else:
        for name in ("fork_safe", "subprocess"):
            got = report[name]
            rc = got["returncode"]
            if rc == 0:
                verdict = "ran" + ("" if got["stdout"] == cwd else f", but in {got['stdout']!r}")
            elif rc in (10, 11, -10, -11):
                verdict = "died of a fault before exec"
            else:
                verdict = got.get("error") or f"exit status {rc}"
            print(f"{name:10s} returncode={rc!s:5s} {verdict}")
        print(f"(pyproj {report['pyproj']}, PROJ {report['proj']}, {args.objects} objects freed)")
    return 0 if report["fork_safe"]["returncode"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
