"""Sandboxed code execution for the agents (container-per-run).

Lets the code / analysis agents **run and debug** code in an isolated sandbox.
The default backend launches a **fresh, hardened container per run**:

  docker run --rm --network none --read-only --cap-drop ALL
    --security-opt no-new-privileges --memory … --cpus … --pids-limit …
    --tmpfs /tmp --user <host-uid> -v <workdir>:/work:rw <image> python /work/script.py

i.e. no network, read-only root filesystem, dropped capabilities, no privilege
escalation, CPU/memory/pid limits, a wall-clock timeout, and the *only* writable
location is a throwaway work dir whose files are persisted as output artifacts.

A ``local`` subprocess backend exists for development only — it is **NOT a
sandbox** (it runs on the host) and must be opted into explicitly.

Gated by ``AGENT_CODE_EXEC``, which defaults to ON. Treat all executed code as
untrusted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime.extraction_flag import extraction_enabled
# Children are started through fork_safe, never subprocess directly: on macOS a fork of this
# process after its first reprojection kills the child before exec (see fork_safe).
from agent_runtime import fork_safe

def _num_env(name: str, default: float) -> float:
    """A numeric env var that tolerates being present but blank.

    `.env` files and compose interpolation both produce ``KEY=`` for an unset optional value,
    and a bare ``int(os.getenv(...))`` on that raises at IMPORT time — which does not degrade
    the sandbox, it takes the whole process down before it serves a request.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# NOTE the default is the SLIM image on purpose: a fresh checkout must be able to run code
# without first building anything. The pre-baked image is opt-in via the env var, and the
# probe below reports which one is actually in play — see _log_image_choice.
DEFAULT_IMAGE = os.getenv("AGENT_CODE_EXEC_IMAGE", "python:3.11-slim")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_CODE_EXEC_TIMEOUT", "60"))
DEFAULT_INSTALL_TIMEOUT = int(os.getenv("AGENT_CODE_EXEC_INSTALL_TIMEOUT", "300"))
# Per-run sandbox budget. 512m was too small for the real work this agent is asked to do:
# a city-scale incident CSV (~130k rows) through pandas + geopandas, a KDE/hexbin heatmap, or
# a spatial join all exceed it and die as an OOM kill with no useful stderr. 4g is generous
# for that class of job while still bounding a runaway loop far below a real server's RAM —
# raise AGENT_CODE_EXEC_MEMORY on a big host (a 60 GB box can comfortably afford 8g-16g).
DEFAULT_MEMORY = os.getenv("AGENT_CODE_EXEC_MEMORY", "4g")
# The deps-install phase needs headroom of its own: building/installing the scientific stack
# (numpy/pandas/scipy/geopandas) was getting OOM-killed (exit 137) under the old exec limit.
DEFAULT_INSTALL_MEMORY = os.getenv("AGENT_CODE_EXEC_INSTALL_MEMORY", "2g")
# 1 CPU serializes pandas/geopandas work that is trivially parallel; 2 keeps a single run
# responsive without letting one job monopolize a shared host.
DEFAULT_CPUS = os.getenv("AGENT_CODE_EXEC_CPUS", "2.0")
DEFAULT_PIDS = os.getenv("AGENT_CODE_EXEC_PIDS", "256")
MAX_OUTPUT_CHARS = 20_000
MAX_ARTIFACTS = 20
# Above this, an output file is called out in the run result. Cost was invisible: a 37 MB CSV
# was converted into an 89 MB intermediate GeoJSON and then immediately downsampled, every
# turn, with nothing in the transcript hinting that it had happened.
LARGE_ARTIFACT_MB = float(os.getenv("AGENT_LARGE_ARTIFACT_MB", "25"))
MAX_DEPS = 50
# Deps install under this dir inside the work dir; added to PYTHONPATH for the run.
DEPS_DIRNAME = ".deps"
# pip's scratch space during install — kept on the host-backed work dir (real disk)
# rather than the small in-container tmpfs, so installing big wheels (numpy/matplotlib)
# doesn't run out of space.
PIPTMP_DIRNAME = ".piptmp"

# A conversation's durable workspace is never reaped otherwise: one directory per thread,
# forever, including every one-off question. That was survivable while a workspace held a few
# output files; it is not once it also holds a site-packages tree.
WORKSPACE_TTL_HOURS = _num_env("AGENT_CODE_EXEC_WS_TTL_HOURS", 72.0)
# Above this, the conversation's dependency cache is dropped and rebuilt. A cache is an
# optimisation, so the safe failure is to lose it, never to fill the disk.
DEPS_CACHE_MAX_MB = int(_num_env("AGENT_CODE_EXEC_DEPS_CACHE_MB", 2048))
# How often the TTL sweep may actually walk the work root (see _sweep_workspaces).
SWEEP_INTERVAL_S = 600.0
_last_sweep = 0.0
# When set, per-run sandbox work dirs are created under this path instead of the
# system temp dir. This is REQUIRED for Docker-out-of-Docker (agent runs in a
# container, shelling out to the host daemon): the path must exist at the SAME
# absolute location on the host (a shared bind mount), so the `docker run -v <work>:/work`
# the agent issues resolves to a real host directory the daemon can mount.
WORK_ROOT_ENV = "AGENT_CODE_EXEC_WORK_ROOT"
# The extracted method library is mounted READ-ONLY so the agent can
# `from iguide_methods import ...` and compose validated units in Python instead of chaining
# tool calls. Read-only because the library is generated from ingested elements: a run must
# never be able to edit the thing later runs will trust.
METHOD_LIBRARY_DIRNAME = "method_library"
METHOD_LIBRARY_MOUNT = "/opt/iguide_methods"

# A conservative pip requirement spec: name[extras]version-specifiers. No flags,
# no whitespace, no shell/path characters — deps are passed as argv (never a shell).
_DEP_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([<>=!~]=?[A-Za-z0-9._*+!-]+(,[<>=!~]=?[A-Za-z0-9._*+!-]+)*)?$"
)


# Heavy frameworks that take minutes to install and are rarely needed for the
# geospatial/data tasks this sandbox serves. Rejected unless an explicit
# AGENT_CODE_EXEC_PIP_ALLOW opts them in — this stops an ungrounded request (e.g.
# "forecast with an LSTM") from burning the whole install budget before the agent
# can recover. Tune via AGENT_CODE_EXEC_PIP_DENY (comma-separated, replaces this set).
_DEFAULT_PIP_DENY = {
    "tensorflow", "tensorflow-gpu", "tensorflow-cpu", "torch", "torchvision",
    "torchaudio", "jax", "jaxlib", "transformers", "keras", "mxnet", "paddlepaddle",
}


def _pip_denylist() -> set:
    raw = os.getenv("AGENT_CODE_EXEC_PIP_DENY")
    if raw is None:
        return _DEFAULT_PIP_DENY
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _sanitize_deps(dependencies: Optional[List[Any]]) -> Tuple[List[str], List[str]]:
    """Return (allowed, rejected) pip specs. Rejects flags/odd specs; honors an explicit
    allowlist (AGENT_CODE_EXEC_PIP_ALLOW), else a denylist of heavy frameworks."""
    if not dependencies:
        return [], []
    allow = {x.strip().lower() for x in (os.getenv("AGENT_CODE_EXEC_PIP_ALLOW") or "").split(",") if x.strip()}
    deny = _pip_denylist()
    allowed: List[str] = []
    rejected: List[str] = []
    for raw in dependencies:
        spec = str(raw).strip()
        if not spec or spec.startswith("-") or any(c.isspace() for c in spec) or not _DEP_RE.match(spec):
            rejected.append(spec)
            continue
        name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()
        if allow:
            if name not in allow:
                rejected.append(spec)
                continue
        elif name in deny:
            rejected.append(spec)
            continue
        allowed.append(spec)
    return allowed[:MAX_DEPS], rejected


# The sandbox image ships no third-party packages, so code that imports pandas dies with
# ModuleNotFoundError unless `dependencies` was passed. Observed in every data task: the model
# omits it, burns a container run, reads the traceback, then retries the byte-identical code
# with dependencies set. Infer the obvious ones from the source instead of charging the user a
# failed run for a detail the code already states.
_IMPORT_TO_PIP = {
    "pandas": "pandas", "geopandas": "geopandas", "numpy": "numpy", "shapely": "shapely",
    "matplotlib": "matplotlib", "scipy": "scipy", "sklearn": "scikit-learn", "pyproj": "pyproj",
    "fiona": "fiona", "rasterio": "rasterio", "seaborn": "seaborn", "statsmodels": "statsmodels",
    "pyarrow": "pyarrow", "networkx": "networkx", "folium": "folium", "mapclassify": "mapclassify",
    "requests": "requests", "bs4": "beautifulsoup4", "PIL": "pillow", "openpyxl": "openpyxl",
}


def _infer_deps(code: str, declared: List[str]) -> List[str]:
    """pip names for third-party modules the code imports but nobody declared."""
    imported = set(re.findall(r"^[ \t]*(?:import|from)[ \t]+([A-Za-z_][\w.]*)", str(code or ""), re.M))
    tops = {name.split(".")[0] for name in imported}
    have = {re.split(r"[<>=!~\[]", d)[0].strip().lower() for d in declared}
    return [pip for mod, pip in _IMPORT_TO_PIP.items() if mod in tops and pip.lower() not in have]


# ---------------------------------------------------------------------------
# What the sandbox image already ships
# ---------------------------------------------------------------------------
# ``pip install --target /work/.deps`` installs UNCONDITIONALLY — it does not consult the
# image's own site-packages. So baking pandas into the sandbox image saves nothing on its own:
# the install phase still runs and still spends its budget. The executor closes that gap by
# asking the image what it can already import and dropping those from the install list.
#
# The set is PROBED rather than declared, because a declared list drifts: an image that stops
# shipping a package would then suppress an install the code needs, and the run would die on
# ModuleNotFoundError with nothing naming the cause. A probe that fails returns nothing, so the
# worst case is the install phase we already pay for today.
PREINSTALLED_ENV = "AGENT_CODE_EXEC_PREINSTALLED"
PROBE_TIMEOUT = int(_num_env("AGENT_CODE_EXEC_PROBE_TIMEOUT", 60))

_LOG = logging.getLogger(__name__)
_probe_cache: Dict[str, frozenset] = {}
_image_logged: set = set()
_probe_lock = threading.Lock()

# A REAL import, not importlib.util.find_spec. find_spec locates the module file without
# loading it, so it answers True for a package whose extension module cannot link — measured:
# fiona and rasterio in a slim-based image find fine and then die on
# `ImportError: libexpat.so.1`. Skipping the install on the strength of that turns a slow run
# into a broken one, which is precisely the drift this probe exists to prevent. Importing all
# twenty costs ~2.5 s, once per process.
# Reports {pip_name: version}. The VERSION matters as much as the presence: see
# _constraints_text — an install into --target re-resolves the whole closure, so without the
# image's own versions to pin against it silently replaces them.
_PROBE_SCRIPT = (
    "import importlib, json, sys\n"
    "try:\n"
    "    from importlib.metadata import version as _v\n"
    "except Exception:\n"
    "    _v = lambda n: ''\n"
    "out = {}\n"
    "for mod, pip in json.loads(sys.argv[1]).items():\n"
    "    try:\n"
    "        importlib.import_module(mod)\n"
    "    except Exception:\n"
    "        continue\n"
    "    try:\n"
    "        out[pip] = _v(pip)\n"
    "    except Exception:\n"
    "        out[pip] = ''\n"
    "print(json.dumps(out))\n"
)


def _constraints_text(versions: Dict[str, str]) -> str:
    """A pip constraints file pinning the image's own packages to the image's own versions.

    ``pip install --target`` sets ignore_installed=True, so it re-resolves the FULL transitive
    closure of whatever is asked for — including numpy and pandas that the image already has —
    and installs them into the cache. PYTHONPATH precedes site-packages, so those copies then
    win for the rest of the conversation: asking for one small package silently swaps the
    numpy underneath rasterio and geopandas, which were compiled against the image's. The
    model is told ``installed: []`` for numpy, so nothing in the transcript shows the swap.

    Constraints do not install anything; they only bound versions if pip pulls one in. A
    genuine conflict now fails loudly with pip naming it, which beats an ABI mismatch that
    surfaces as a segfault three tools later.
    """
    if (os.getenv("AGENT_CODE_EXEC_PIN_IMAGE", "1") or "").strip().lower() in {"0", "false", "no", "off"}:
        # The escape hatch is real: pinning every baked package also blocks a legitimate
        # request for a newer one. Turning it off restores the silent-swap behaviour, which
        # is the right trade only if you know the image's compiled packages tolerate it.
        return ""
    lines = [f"{name}=={ver}" for name, ver in sorted(versions.items()) if ver]
    return "\n".join(lines) + ("\n" if lines else "")


def _dep_name(spec: str) -> Optional[str]:
    """The bare pip name of *spec*, or None when it carries a version or extra constraint.

    A pinned spec (``geopandas==0.14``) must not be satisfied by whatever version the image
    happens to ship, so only unconstrained names are eligible to be skipped.
    """
    text = str(spec or "").strip()
    if not text or re.search(r"[<>=!~\[]", text):
        return None
    return text.lower()


def _preinstalled_override() -> Optional[frozenset]:
    """``AGENT_CODE_EXEC_PREINSTALLED`` as a set, or None when unset.

    Set it to a comma-separated list to skip the probe (a custom backend, or a test), or to
    the empty string to turn the optimisation off entirely.
    """
    raw = os.getenv(PREINSTALLED_ENV)
    if raw is None:
        return None
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def _drop_preinstalled(deps: List[str], have: frozenset) -> Tuple[List[str], List[str]]:
    """Split *deps* into (still to install, already in the image)."""
    if not have:
        return list(deps), []
    have_norm = {_normalize_dist(h) for h in have}
    kept: List[str] = []
    skipped: List[str] = []
    for spec in deps:
        name = _dep_name(spec)
        hit = bool(name) and _normalize_dist(name) in have_norm
        (skipped if hit else kept).append(spec)
    return kept, skipped


# A run phase WITH network, for measurement only. The comparison of 2026-10-09 (stage
# "Three ways to reach public data") needed the same peer with one difference, the run
# container's network, so this flag changes the run phase's `--network` and nothing else.
# Refused unless AGENT_MODE is explicitly `local` or `dev`: on any other mode, and with
# AGENT_MODE unset (which defaults to dev), the run stays `--network none` and a warning
# says why. A networked run reaches whatever its host reaches (the services beside it, the
# cloud metadata endpoint, the LAN), so it is never a deployment setting.
EXEC_NETWORK_ENV = "AGENT_CODE_EXEC_NETWORK"
_EXEC_NETWORK_MODES = ("local", "dev")


def exec_network() -> Optional[str]:
    """The docker network for the run phase, or None for ``--network none``.

    ``AGENT_CODE_EXEC_NETWORK``: unset/0/false/off/none = no network; 1/true/on = the default
    ``bridge``; any other value names a docker network.
    """
    raw = (os.getenv(EXEC_NETWORK_ENV) or "").strip()
    if raw.lower() in ("", "0", "false", "off", "no", "none"):
        return None
    mode = (os.getenv("AGENT_MODE") or "").strip().lower()
    if mode not in _EXEC_NETWORK_MODES:
        _LOG.warning("%s=%s REFUSED: AGENT_MODE must be set to one of %s (it is %r); the "
                       "sandbox run phase stays --network none", EXEC_NETWORK_ENV, raw,
                       "/".join(_EXEC_NETWORK_MODES), mode or "unset")
        return None
    if raw.lower() in ("1", "true", "on", "yes"):
        return "bridge"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", raw):
        _LOG.warning("%s=%r is not a docker network name; the run stays --network none",
                       EXEC_NETWORK_ENV, raw)
        return None
    return raw


# Told to the model only while the run phase really has network: every prompt and tool
# description here says the sandbox has none, and a model told so never tries.
EXEC_NETWORK_NOTE = (
    "NETWORK (this deployment only): the code's RUN phase HAS outbound network. Code may "
    "download public data itself (requests, urllib, geopandas.read_file(url), pystac-client, "
    "rasterio on an https URL). This overrides any instruction elsewhere that says the "
    "sandbox has no network. Credentials are not available inside the sandbox. "
)


def sandbox_capability_note() -> str:
    """What the code can reach beyond its files, for the peers' prompts and execute_code."""
    from agent_runtime import tool_bridge

    note = EXEC_NETWORK_NOTE if exec_network() else ""
    if tool_bridge.enabled():
        note += tool_bridge.model_note()
    return note


def is_code_exec_enabled() -> bool:
    """Whether code execution is enabled. **On by default**; set ``AGENT_CODE_EXEC`` to a falsy
    value (0/false/no/off) to disable the sandboxed ``execute_code`` tool."""
    return (os.getenv("AGENT_CODE_EXEC", "1") or "").strip().lower() not in {"0", "false", "no", "off"}


def _clip(text: Any, limit: int = MAX_OUTPUT_CHARS) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else (s[:limit] + f"\n…[truncated {len(s) - limit} chars]")


def method_library_root() -> Optional[Path]:
    """Where the generated method library lives, built or not: the ONE answer for its reader and
    its writers.

    They used to disagree. The reader (below) honoured ``AGENT_METHOD_LIBRARY_DIR`` while both
    writers — ``extractors/emitters/library_emitter.py`` and ``scripts/build_method_library.py``
    — wrote to ``storage_root()/method_library`` regardless. Setting the variable, which
    Docker-out-of-Docker requires (docker-compose.extraction.yml), therefore built the library in
    one place and read it from another, empty one.
    """
    override = (os.getenv("AGENT_METHOD_LIBRARY_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from agent_runtime.file_store import storage_root
        return Path(storage_root()) / METHOD_LIBRARY_DIRNAME
    except Exception:
        return None


def method_library_dir() -> Optional[Path]:
    """Path of the generated method library, or None when nothing is ingested yet.

    Under Docker-out-of-Docker this path is also the bind SOURCE of the sandbox's library mount,
    which the host's daemon resolves, so it must exist at the same absolute path on the host.
    """
    p = method_library_root()
    # "Ingested" means the package exists, not merely the directory. An empty directory used to
    # count, so it was mounted and put first on the sandbox PYTHONPATH with nothing in it.
    return p if (p is not None and (p / "iguide_methods").is_dir()) else None


def artifacts_enabled() -> bool:
    from agent_runtime.artifacts import artifacts_enabled as _enabled
    return _enabled()


def invariant_gate_enabled() -> bool:
    """Whether to append the in-sandbox invariant checks. **On whenever the extraction bundle is.**

    Unset, ``AGENT_INVARIANT_GATE`` follows ``AGENT_EXTRACTION`` (agent_runtime/extraction_flag.py),
    which is off by default, so a deployment that has not opted into the bundle runs exactly the
    code it ran before. Inside the bundle, on-by-default is deliberate: the failure it catches — a
    distance computed in a geographic CRS — produces a plausible number and no error, so a gate
    that has to be remembered protects nobody. An explicit value wins either way. The epilogue
    cannot fail a run (every check is guarded and the writer swallows OSError), so the cost of
    leaving it on is one JSON file.
    """
    raw = (os.getenv("AGENT_INVARIANT_GATE") or "").strip().lower()
    if not raw:
        # Unset: follow the extraction bundle (agent_runtime/extraction_flag.py), OFF by default,
        # so integrating it changes no deployed run until it is switched on.
        return extraction_enabled()
    return raw not in {"0", "false", "no", "off"}


def contracts_for_code(code: str) -> Dict[str, Any]:
    """Declared invariants for every library unit this code imports.

    Resolved AGENT-side from the registry, then injected into the run as a literal. The sandbox
    has no network and the library mount is optional, so a guard that tried to read the registry
    itself could silently fail to install — and a guard that does not install reads exactly like
    a contract that passed.
    """
    try:
        from agent_runtime.artifacts import library_units_used
        from agent_runtime.method_library import load_registry
    except Exception:
        return {}
    used = library_units_used(code or "")
    if not used:
        return {}
    registry = load_registry()
    if not registry:
        return {}
    out: Dict[str, Any] = {}
    for ref in used:
        module, symbol = ref.get("module"), ref.get("symbol")
        if not (module and symbol):
            continue
        # Match on the ELEMENT PACKAGE plus symbol, not on the exact `v_<sha>` module string.
        # Exact-string equality meant the element-package alias
        # (`from iguide_methods.ke_x import symbol`) resolved nothing, so importing a unit the
        # documented friendly way turned enforcement off silently.
        element_pkg = module.split(".")[1] if module.count(".") >= 1 else ""
        entry = next((v for v in registry.values()
                      if isinstance(v, dict) and v.get("module") == module
                      and v.get("library_symbol") == symbol), None)
        if entry is None and element_pkg:
            entry = next((v for v in registry.values()
                          if isinstance(v, dict) and not v.get("ambiguous")
                          and v.get("library_symbol") == symbol
                          and v.get("element_package") == element_pkg), None)
        invariants = [i for i in ((entry or {}).get("invariants") or []) if isinstance(i, dict)]
        if invariants:
            # Keyed and patched on the REGISTRY's module, which is the one that actually holds
            # the function object — the alias path resolves to the same unit.
            out[symbol] = {"module": (entry or {}).get("module") or module,
                           "symbol": symbol, "invariants": invariants}
    return out


def _note_printed_outputs(verification: Dict[str, Any], stdout: Any, work: Path) -> Dict[str, Any]:
    """Name the near-miss: ``print('IGUIDE_OUTPUTS =', {...})`` declares nothing.

    The gate reads the module-level VARIABLE, so a run that printed its numbers instead declared
    none, and none of them was checked. Seen in the map UI on 2026-10-01: ``declared_outputs.json``
    was ``{}`` on a run whose stdout carried the dict. Only the agent side sees stdout, so it is
    detected here and stated as a cannot_determine that says what to do. Never raises.
    """
    try:
        from agent_runtime.sandbox_verify import DECLARED_FILENAME, DECLARED_OUTPUTS, UNKNOWN

        if not verification or DECLARED_OUTPUTS not in str(stdout or ""):
            return verification
        path = work / DECLARED_FILENAME
        declared = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if declared:
            return verification
        finding = {"check": "declared_outputs", "status": UNKNOWN, "target": DECLARED_OUTPUTS,
                   "message": (f"{DECLARED_OUTPUTS} was printed, not assigned: the gate reads a "
                               f"module-level variable, so none of the printed numbers was "
                               f"checked. Assign {DECLARED_OUTPUTS} = {{...}} at module scope.")}
        counts = dict(verification.get("counts") or {})
        counts[UNKNOWN] = counts.get(UNKNOWN, 0) + 1
        verdict = verification.get("verdict")
        return {**verification, "findings": [*(verification.get("findings") or []), finding],
                "counts": counts, "verdict": verdict if verdict == "fail" else UNKNOWN}
    except Exception:
        return verification


def _read_checks(work: Path) -> Dict[str, Any]:
    """Load the invariant gate's report, if it wrote one.

    Absent is not a failure: the gate may be disabled, or the run may have died before the
    epilogue. Absent and "checked, all fine" must stay distinguishable, so this returns {} for
    absent rather than a synthetic pass.
    """
    from agent_runtime.sandbox_verify import CHECKS_FILENAME

    path = work / CHECKS_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Keep the payload small, but never at the cost of the evidence for the verdict. The cap
    # used to slice an unsorted list, so a report with 12 cannot_determine findings ahead of a
    # single `fail` shipped verdict="fail" with no fail finding — and the gate keyed on
    # findings, so nothing reacted.
    # Units are judged HERE, where a unit library is available, from the declarations the
    # sandbox reported as written (stage 43; agent_runtime/declared_outputs.py). The typed
    # values travel on the result as `outputs`: {name, value, unit, dimension, measured_in_crs}.
    outputs: List[Dict[str, Any]] = []
    try:
        from agent_runtime import declared_outputs

        extra, outputs = declared_outputs.evaluate(data)
        data = {**data, "findings": [*(data.get("findings") or []), *extra],
                **declared_outputs.merge_verdict(data, extra)}
    except Exception as exc:  # noqa: BLE001 - the unit check must never cost the run its report
        # Never silently: an image built without the unit library would otherwise pass a null
        # unit. Every declaration becomes an explicit unknown.
        if data.get("declared"):
            extra = [{"check": "declared_units", "status": "cannot_determine",
                      "target": str(d.get("name")),
                      "message": f"declared units were not checked: {type(exc).__name__}: {exc}"}
                     for d in data["declared"] if isinstance(d, dict)][:12]
            counts = dict(data.get("counts") or {})
            counts["cannot_determine"] = counts.get("cannot_determine", 0) + len(extra)
            data = {**data, "findings": [*(data.get("findings") or []), *extra],
                    "counts": counts,
                    "verdict": "fail" if counts.get("fail") else "cannot_determine"}
    order = {"fail": 0, "cannot_determine": 1}
    findings = sorted((f for f in (data.get("findings") or [])
                       if isinstance(f, dict) and f.get("status") != "pass"),
                      key=lambda f: order.get(f.get("status"), 2))
    truncated = max(0, len(findings) - 12)
    return {"verdict": data.get("verdict"), "counts": data.get("counts") or {},
            "inspected": data.get("inspected") or [], "findings": findings[:12],
            **({"outputs": outputs} if outputs else {}),
            **({"metric_ops": data["metric_ops"][:12]} if data.get("metric_ops") else {}),
            **({"findings_truncated": truncated} if truncated else {}),
            **({"error": data["error"]} if data.get("error") else {})}


def session_workspace_dir(session: Optional[str]) -> Optional[Path]:
    """The durable workspace for *session*, or None without a work root.

    Public so agent-side staging writes into the SAME directory the next run carries into /work:
    a file staged anywhere else would exist on the host and be missing in the container, which
    surfaces inside a sandbox with no network as a bare "file not found".
    """
    return _session_workspace(session)


@dataclass
class ExecResult:
    exit_code: Optional[int]
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = ""
    code: str = ""   # the executed source (also saved as a downloadable artifact)
    installed: List[str] = field(default_factory=list)  # pip deps installed before the run
    # Findings from the in-sandbox invariant gate. Empty dict = the gate did not run.
    verification: Dict[str, Any] = field(default_factory=dict)
    # Calls the code made through the tool bridge (agent_runtime/tool_bridge.py), in order.
    bridge_calls: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "error": self.error,
            "code": self.code,
            "installed": self.installed,
            "artifacts": self.artifacts,
            "backend": self.backend,
            # Surfaced INSIDE the tool result, not appended to the answer afterwards, so the
            # model can react in-loop: reproject and re-run rather than caveat a wrong number.
            "verification": self.verification,
            **({"bridge_calls": self.bridge_calls} if self.bridge_calls else {}),
        }


def _persist_artifacts(work: Path, exclude: set) -> List[Dict[str, Any]]:
    """Persist files the run created in *work* to the agent file store."""
    from agent_runtime.file_store import create_output_file_from_path

    artifacts: List[Dict[str, Any]] = []
    for path in sorted(work.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(work)
        if str(rel) in exclude or (rel.parts and rel.parts[0] in {"__pycache__", DEPS_DIRNAME, PIPTMP_DIRNAME}):
            continue
        if len(artifacts) >= MAX_ARTIFACTS:
            break
        try:
            rec = create_output_file_from_path(path, filename=path.name)
            artifacts.append(
                {
                    "file_id": rec["file_id"],
                    "filename": rec["filename"],
                    "download_url": rec.get("download_url"),
                    "size_bytes": rec.get("size_bytes"),
                }
            )
        except Exception:
            continue
    return artifacts


# A container killed by a signal exits with a NEGATIVE code and usually writes nothing to
# stderr. Reported bare, that reads to the model as "it just failed", and it then invents a
# cause (observed: "failed due to dependency issues") and gives up instead of retrying. Naming
# the signal — and what usually causes it here — lets the agent choose a real next step.
_SIGNAL_DIAGNOSIS = {
    9: ("SIGKILL", "the sandbox hit its memory limit (or was killed). Reduce the data held in "
                   "memory — read in chunks, downsample, or write results incrementally."),
    11: ("SIGSEGV", "the sandbox process crashed. This is usually a native-library or memory "
                    "fault, not your logic; retry once, and if it repeats use smaller inputs or "
                    "avoid the heavy native dependency (a pure-stdlib or pandas-only version "
                    "often works)."),
    6: ("SIGABRT", "a native library aborted the process. Try a simpler approach or fewer "
                   "third-party dependencies."),
    15: ("SIGTERM", "the sandbox was terminated (time or resource limit)."),
}



def _size_report(artifacts: List[Dict[str, Any]]) -> Optional[str]:
    """Name the run's output sizes when they are big enough to matter."""
    sized = [(a.get("filename") or "?", int(a.get("size_bytes") or 0)) for a in artifacts]
    big = [(n, b) for n, b in sized if b >= LARGE_ARTIFACT_MB * 1024 * 1024]
    if not big:
        return None
    total_mb = sum(b for _, b in sized) / 1024 / 1024
    listed = ", ".join(f"{n} {b / 1024 / 1024:.1f} MB" for n, b in big)
    return (f"[large output: {listed}; {total_mb:.1f} MB written this run. If it is an intermediate, "
            f"the geo tools read the ORIGINAL upload directly (CSV/shapefile/GeoPackage/GeoParquet), "
            f"so converting first is usually avoidable; otherwise write only the columns you need.]")


def _diagnose_abnormal_exit(exit_code: Optional[int], stderr: str, error: Optional[str]) -> Optional[str]:
    """Explain a signal-killed run so the caller gets a cause, not a silent failure.

    Two conventions reach us: a negative code when the docker CLI itself is signalled, and
    128+N when the CONTAINER is signalled (docker's own convention — an OOM kill is 137).
    Both used to surface as a bare non-zero exit with empty stderr.
    """
    if error or not isinstance(exit_code, int) or exit_code == 0:
        return None
    if exit_code < 0:
        signo = -exit_code
    elif 128 < exit_code < 160:
        signo = exit_code - 128
    else:
        return None  # an ordinary non-zero exit: the traceback in stderr is the explanation
    name, hint = _SIGNAL_DIAGNOSIS.get(signo, (f"signal {signo}", "the sandbox terminated abnormally."))
    detail = f"code execution was killed by {name} (exit {exit_code}); {hint}"
    if not (stderr or "").strip():
        detail += " No stderr was produced, so nothing was written and no output files exist."
    return detail


# The top-level package only. `No module named 'iguide_methods.ke_x'` names a wrong element
# module, which the traceback already explains; it is not a missing library.
_LIBRARY_MISSING = "No module named 'iguide_methods'"
_library_warned: set = set()


def _diagnose_library_import(stderr: str, backend: str) -> Optional[str]:
    """Say WHY ``iguide_methods`` would not import, which the traceback cannot.

    Three causes raise the same ModuleNotFoundError and call for different responses. The one
    that matters is a library that is built but invisible to the sandbox. Under
    Docker-out-of-Docker the host's daemon resolves the bind source, and the default location
    (storage_root()/method_library) sits on a named volume the host does not have at that path,
    so the daemon creates an empty directory and mounts THAT. Meanwhile the agent side lists the
    methods, with import lines, and every one of them fails here.
    """
    if _LIBRARY_MISSING not in (stderr or ""):
        return None
    if not extraction_enabled():
        return "[`iguide_methods` does not exist in this deployment; write the function inline.]"
    lib = method_library_dir()
    if lib is None:
        return ("[no method library has been built in this deployment, so `iguide_methods` does "
                "not exist yet; write the function inline.]")
    if str(lib) not in _library_warned:
        _library_warned.add(str(lib))
        _LOG.warning(
            "method library at %s is built but a %s sandbox could not import it.%s", lib, backend,
            " Under Docker-out-of-Docker the daemon resolves that path on the HOST, so it must "
            "exist at the same absolute path there (docker-compose.extraction.yml); otherwise an "
            "empty directory is mounted." if backend == "docker" else "")
    return ("[the method library is built in this deployment but is not visible inside the "
            "sandbox: a mount problem on the server, logged for its operator. No library method "
            "can be imported until it is fixed; write the function inline.]")


def _describe_code(code: str) -> Optional[str]:
    """A short slug for what a script does, read from the code itself.

    Several execute_code calls in one turn all produced ``executed_code.py``, so the
    download list showed the same name three or four times with no way to tell which run
    was which. Prefer the module docstring / first comment (what the author said it does),
    then the first function name; give up rather than invent something meaningless.
    """
    text = str(code or "")
    m = re.search(r'^\s*(?:"""|\'\'\')\s*(.+)', text) or re.search(r"^\s*#\s*(.+)", text, re.M)
    if not m:
        m = re.search(r"^\s*def\s+([A-Za-z_]\w*)", text, re.M)
    if not m:
        return None
    words = re.findall(r"[A-Za-z0-9]+", m.group(1).lower())[:5]
    slug = "_".join(words)[:48].strip("_")
    return slug or None


def _persist_source(code: str, *, label: Optional[str] = None,
                    filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """Save the executed source as a downloadable output artifact.

    Named for what the script does (caller-supplied ``label``, else derived from the
    code) so repeated runs in one conversation are distinguishable.
    """
    from agent_runtime.file_store import create_output_file

    if not filename:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "")).strip("_").lower()[:48]
        stem = stem or _describe_code(code) or "executed_code"
        filename = f"{stem}.py"
    try:
        rec = create_output_file(filename, code or "")
        return [{
            "file_id": rec["file_id"],
            "filename": rec["filename"],
            "download_url": rec.get("download_url"),
            "size_bytes": rec.get("size_bytes"),
            "kind": "source",
        }]
    except Exception:
        return []


def _stage_inputs(work: Path,
                  input_files: Optional[List[Dict[str, str]]]
                  ) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """Copy requested input files into *work* so the sandboxed code can read them.

    Each spec is ``{"source": <host_path>, "dest": <relative_name>}``.  ``dest`` must
    be a plain filename (no path separators / traversal) that stays inside *work* —
    *work* is the only writable mount, so files placed here appear at ``/work/<dest>``
    inside the container (the run's working directory).

    Returns ``(staged_dest_names, errors, shadowed)`` — ``shadowed`` being names where an
    upload landed on top of a file the conversation's workspace already had.
    """
    staged: List[str] = []
    errors: List[Dict[str, str]] = []
    shadowed: List[str] = []
    work_resolved = work.resolve()
    for spec in input_files or []:
        source = str((spec or {}).get("source") or "").strip()
        dest = str((spec or {}).get("dest") or "").strip()
        if not source or not dest:
            continue
        if "/" in dest or "\\" in dest or dest in {".", ".."}:
            errors.append({"dest": dest, "error": "invalid destination name"})
            continue
        target = (work / dest).resolve()
        if target.parent != work_resolved:
            errors.append({"dest": dest, "error": "destination escapes work dir"})
            continue
        src = Path(source).expanduser()
        if not src.is_file():
            errors.append({"source": source, "error": "source file not found"})
            continue
        try:
            # Uploads are staged AFTER the workspace is carried in, so an upload whose
            # filename matches a file the model wrote wins silently — it reads its own
            # edited file and gets the original upload. Staging still wins (the user's data
            # must be reachable under its own name), but the run is told.
            if target.exists():
                shadowed.append(dest)
            shutil.copyfile(src, target)
            staged.append(dest)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"source": source, "dest": dest, "error": f"{type(exc).__name__}: {exc}"})
    return staged, errors, shadowed


def _work_root() -> Optional[str]:
    """Directory under which per-run work dirs are created (None = system temp).

    Set via ``AGENT_CODE_EXEC_WORK_ROOT`` for Docker-out-of-Docker so the work dir
    sits on a host-shared path the host daemon can bind-mount.
    """
    root = (os.getenv(WORK_ROOT_ENV) or "").strip()
    if not root:
        return None
    try:
        os.makedirs(root, exist_ok=True)
        return root
    except OSError:
        return None


def _session_workspace(session: Optional[str]) -> Optional[Path]:
    """The durable working directory for a conversation's code, or None.

    Each run gets a throwaway dir and an ``--rm`` container, so a follow-up run used to
    start from an empty directory: the file the previous step wrote was gone, and the only
    way to continue a piece of work was to re-upload it by file_id. The CONTAINER stays
    ephemeral (that is the sandbox), but the workspace persists per conversation, so
    "now add a heatmap of that GeoJSON" can build on what the last step produced.
    """
    key = str(session or "").strip()
    if not key:
        return None
    root = _work_root() or tempfile.gettempdir()
    # Slug PLUS a digest of the real key. The slug alone is many-to-one — it maps every
    # unsafe character to "_" and then truncates — so thread ids 'sess:42' and 'sess_42',
    # both raw client input, would land in one directory and share its files and its package
    # cache. claude_peer.session_dir carries the same digest for the same reason.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:40]
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    try:
        path = Path(root) / f"agentws_{safe}_{digest}"
        _sweep_workspaces(Path(root))
        path.mkdir(parents=True, exist_ok=True)
        # Mark the workspace as used. A directory's mtime only moves when an entry is added
        # or removed, so a long conversation that keeps rewriting the same files would look
        # untouched to the sweep and be deleted underneath itself.
        os.utime(path, None)
        return path
    except OSError:
        return None


def _sweep_workspaces(root: Path, *, force: bool = False) -> None:
    """Drop conversation workspaces nobody has touched inside the TTL.

    Throttled because the caller is on the hot path: every execute_code and every workspace
    file tool resolves a workspace, and globbing the work root each time is pure waste on a
    deployment with a thousand conversations in it. A TTL measured in hours does not need a
    sweep measured in milliseconds.

    Failure here is never fatal: a full disk is a problem, but so is refusing to run code
    because a stale directory would not delete.
    """
    if WORKSPACE_TTL_HOURS <= 0:
        return
    import time

    global _last_sweep
    now = time.time()
    if not force and (now - _last_sweep) < SWEEP_INTERVAL_S:
        return
    _last_sweep = now
    cutoff = now - WORKSPACE_TTL_HOURS * 3600
    try:
        candidates = list(root.glob("agentws_*"))
    except OSError:
        return
    for path in candidates:
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def _dir_size_mb(directory: Path) -> float:
    """Bytes under *directory*, in MB, counting as much as can be walked.

    rglob is a generator, so an OSError part-way through (an unreadable directory left by an
    OOM-killed install, ESTALE on a bind mount) used to abort the walk and report 0.0 — which
    reads as "well under the cap" and quietly disables it forever. os.walk skips what it
    cannot read and keeps going, so the total is a floor rather than a zero.
    """
    total = 0
    for base, _dirs, files in os.walk(str(directory), onerror=None):
        for name in files:
            try:
                total += os.stat(os.path.join(base, name)).st_size
            except OSError:
                continue
    return total / (1024 * 1024)


def workspace_deps_dir(session: Optional[str], *,
                       workspace: Optional[Path] = None) -> Optional[Path]:
    """The conversation's durable dependency cache, or None when it has no workspace.

    Packages installed by one run stay importable by the next. This is deliberately NOT part
    of the copy-in/copy-out that carries a workspace's files: a site-packages tree is tens of
    thousands of files, and copying it twice per run would cost more than the pip install it
    is meant to replace. It is bind-mounted into the container instead, so the bytes never
    move — which is also why `_copy_tree` can keep skipping ``.deps`` unchanged.
    """
    if workspace is None:
        workspace = _session_workspace(session)
    if workspace is None:
        return None
    cache = workspace / DEPS_DIRNAME
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache


def reset_deps_cache_if_oversized(cache: Path) -> bool:
    """Drop the cache when it outgrows its cap. Returns True when it was reset.

    Only worth checking before an install — the walk is not free, and that is the only moment
    the cache can grow.
    """
    if DEPS_CACHE_MAX_MB <= 0:
        return False
    if _dir_size_mb(cache) <= DEPS_CACHE_MAX_MB:
        return False
    # Rename aside, THEN delete. Another run of this conversation may have this directory
    # bind-mounted into a live container (the analysis peer and the code peer share the
    # ::codeexec session); rmtree in place pulls site-packages out from under a running
    # import, while a rename leaves that container holding the old directory unharmed.
    try:
        doomed = cache.with_name(f"{cache.name}.evict.{uuid.uuid4().hex[:8]}")
        cache.rename(doomed)
    except OSError:
        return False
    shutil.rmtree(doomed, ignore_errors=True)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return True


def _evict_torn_cache(cache: Optional[Path]) -> None:
    """Drop the cache after an install that did not finish.

    pip moves a package tree and its .dist-info into --target as separate steps, so a kill on
    the install timeout or an OOM leaves the cache half-written. Rebuilding a cache costs one
    install; carrying a torn one costs every later run in the conversation, silently.
    """
    if cache is None or not cache.is_dir():
        return
    try:
        doomed = cache.with_name(f"{cache.name}.torn.{uuid.uuid4().hex[:8]}")
        cache.rename(doomed)
    except OSError:
        return
    shutil.rmtree(doomed, ignore_errors=True)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


def _normalize_dist(name: str) -> str:
    """PEP 503 normalisation, so ``scikit_learn`` and ``scikit-learn`` compare equal."""
    return re.sub(r"[-_.]+", "-", str(name or "")).strip().lower()


def cached_dep_names(cache: Optional[Path]) -> frozenset:
    """Distribution names already installed in *cache*, from their ``.dist-info`` dirs."""
    if cache is None:
        return frozenset()
    names = set()
    try:
        for item in cache.iterdir():
            if not item.is_dir():
                continue
            for suffix in (".dist-info", ".egg-info"):
                if not item.name.endswith(suffix):
                    continue
                # RECORD is written at the END of an install, so its presence is the cheap
                # test that the package beside this metadata actually arrived. pip's --target
                # handling moves `numpy/` and `numpy-2.3.0.dist-info/` as SEPARATE entries,
                # across a bind-mount boundary (so a copy, not a rename) — an install killed
                # on the timeout or by the OOM killer leaves metadata with no package, and
                # trusting the name alone would suppress the reinstall forever.
                if not (item / "RECORD").is_file():
                    continue
                names.add(_normalize_dist(item.name[: -len(suffix)].rsplit("-", 1)[0]))
    except OSError:
        return frozenset()
    return frozenset(names)


def session_workspace_listing(session: Optional[str], *, limit: int = 25) -> List[Dict[str, Any]]:
    """What earlier runs in this conversation left behind: ``[{name, size_bytes}]``.

    A durable workspace is only useful if the model knows what is in it. Without this,
    a steer like "now do a heatmap of that" makes the peer rebuild the dataset it already
    has on disk — or claim it cannot, because nothing told it the file is there.
    """
    ws = _session_workspace(session)
    if ws is None:
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws)
        if rel.parts and rel.parts[0] in {"__pycache__", DEPS_DIRNAME, PIPTMP_DIRNAME}:
            continue
        try:
            items.append({"name": str(rel), "size_bytes": p.stat().st_size})
        except OSError:
            continue
        if len(items) >= limit:
            break
    return items


# Directories inside a workspace that belong to the machinery, not to the conversation.
# Writing into .deps would let one run leave a shadowing module for the next one to import,
# which is the same durable-poisoning path the read-only cache mount closes.
RESERVED_DIRS = {DEPS_DIRNAME, PIPTMP_DIRNAME, "__pycache__"}


def resolve_workspace_file(session: Optional[str], path: str) -> Path:
    """A host path inside this conversation's workspace, or ValueError explaining why not.

    The peer edits files here between runs, so this is the boundary that keeps "patch line 40
    of main.py" from becoming "write anywhere the agent process can reach".
    """
    workspace = _session_workspace(session)
    if workspace is None:
        raise ValueError(
            "this conversation has no durable working directory, so there is no file to edit; "
            "pass the whole program to execute_code as `code` instead")
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("no file named; pass a path relative to the working directory, e.g. 'main.py'")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"{raw!r} is an absolute path; use a path relative to the working directory")
    root = workspace.resolve()
    target = (root / candidate).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{raw!r} points outside the working directory")
    rel = target.relative_to(root)
    # Compared case-INSENSITIVELY: Path.resolve() does not canonicalise case, so on a
    # case-insensitive filesystem (macOS, and a Windows bind mount) '.DEPS/numpy.py' resolves
    # with parts[0] == '.DEPS', slips an exact-match check, and lands in the cache anyway.
    if rel.parts and rel.parts[0].lower() in RESERVED_DIRS:
        raise ValueError(
            f"{rel.parts[0]!r} holds this conversation's installed packages and is not writable; "
            "put your code in a file at the top level, e.g. 'main.py'")
    return target


def _stat_map(directory: Path) -> Dict[str, Tuple[int, int]]:
    """(size, mtime_ns) per relative path — used to tell new/changed files from carried ones."""
    out: Dict[str, Tuple[int, int]] = {}
    for p in directory.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                out[str(p.relative_to(directory))] = (st.st_size, st.st_mtime_ns)
            except OSError:
                continue
    return out


def _copy_tree(src: Path, dst: Path, *, skip: Optional[set] = None) -> None:
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(src))
        if (skip and rel in skip) or rel.split(os.sep)[0] in {"__pycache__", DEPS_DIRNAME, PIPTMP_DIRNAME}:
            continue
        target = dst / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        except OSError:
            continue


def _host_user() -> Optional[str]:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if callable(getuid) and callable(getgid):
        return f"{getuid()}:{getgid()}"
    return None


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------

class CodeExecutor:
    """Base: write code to a throwaway dir, run it, persist artifacts, clean up."""

    backend = "base"

    def _log_image_choice(self, names: frozenset) -> None:
        """Say once which sandbox image is in play, and what it costs.

        The pre-baked image shipped, was built, and was then inert in production for two days:
        the deployment still set the slim default, so every session re-installed the scientific
        stack into its own .deps tree (measured: 22 workspaces, 454 MB) while a 1.24 GB
        iguide-codeexec image sat unused on the same host. Nothing said so — the install phase
        succeeding looks identical to not needing one. This makes the choice legible.

        Takes the resolved probe result rather than reading ``_probe_cache``: called before the
        probe populates it, this warned that a correctly-baked image had no stack at all.
        """
        if self.image in _image_logged:
            return
        _image_logged.add(self.image)
        if names:
            _LOG.info("code sandbox image %s: %d packages pre-installed, no install phase "
                      "needed for them", self.image, len(names))
        else:
            _LOG.warning(
                "code sandbox image %s has no pre-installed scientific stack, so every session "
                "pays a pip install for pandas/geopandas/etc. Build the baked image and point "
                "AGENT_CODE_EXEC_IMAGE at it: docker build -t iguide-codeexec sandbox/",
                self.image)

    def preinstalled(self) -> frozenset:
        """pip names importable in this sandbox with no install phase.

        Empty by default: a backend that cannot verify what it ships must not be able to
        suppress an install, because the failure would surface as ModuleNotFoundError inside
        the user's code rather than as a sandbox problem.
        """
        override = _preinstalled_override()
        return override if override is not None else frozenset()

    def preinstalled_versions(self) -> Dict[str, str]:
        """{pip name: version} for what this sandbox ships. Empty means "not known"."""
        return {}

    def execute(self, code: str, *, language: str = "python", timeout: Optional[int] = None,
                env: Optional[Dict[str, str]] = None,
                dependencies: Optional[List[Any]] = None,
                input_files: Optional[List[Dict[str, str]]] = None,
                label: Optional[str] = None,
                session: Optional[str] = None,
                entrypoint: Optional[str] = None) -> ExecResult:
        if (language or "python").lower() != "python":
            return ExecResult(exit_code=None, error=f"unsupported language: {language}",
                              backend=self.backend, code=(code or ""))
        timeout = int(timeout or DEFAULT_TIMEOUT)
        entrypoint = (str(entrypoint).strip() if entrypoint else "") or None
        if not (code or "").strip() and not entrypoint:
            return ExecResult(
                exit_code=None, backend=self.backend, code="",
                error=("nothing to run: pass `code` with the program to run, or `entrypoint` "
                       "naming a file already in the working directory"))
        if entrypoint and (code or "").strip():
            # Both would silently mean "run the file, ignore the code" — and the code would
            # still be saved as this run's source, so the transcript would show a program that
            # never ran next to output that came from a different one.
            return ExecResult(
                exit_code=None, backend=self.backend, code=(code or ""),
                error=("pass either `code` (run this program) or `entrypoint` (run a file "
                       "already in the working directory), not both. To run edited code, "
                       "write it with write_workspace_file first, then pass `entrypoint`."))

        workspace = _session_workspace(session)
        deps_cache = workspace_deps_dir(session, workspace=workspace)
        # Evict BEFORE deciding what to install, never after: the decision below drops every
        # package the cache already holds, so an eviction later in the run would delete
        # exactly the packages nothing is going to install any more.
        if deps_cache is not None:
            reset_deps_cache_if_oversized(deps_cache)

        # An entrypoint run has no inline source, so infer its imports from the file itself —
        # otherwise re-running a script would install nothing and die on ModuleNotFoundError.
        source_for_deps = code or ""
        if entrypoint:
            # Validated, and the failure is REPORTED rather than swallowed. This is the path
            # that actually runs: an unchecked `entrypoint` of '../x.py' or '/etc/x.py' is a
            # way out of the workspace, and inferring deps from a file we refused to resolve
            # would have run it anyway.
            try:
                resolved = resolve_workspace_file(session, entrypoint)
            except ValueError as exc:
                return ExecResult(exit_code=None, backend=self.backend, code="", error=str(exc))
            entrypoint = resolved.relative_to(workspace.resolve()).as_posix()
            try:
                source_for_deps = resolved.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source_for_deps = ""

        deps, rejected = _sanitize_deps(dependencies)
        auto = _infer_deps(source_for_deps, deps)
        if auto:
            deps = [*deps, *auto]
        # Anything the image already ships is free; installing it again into /work/.deps would
        # shadow the baked copy with an identical one and charge the run for the privilege.
        have = self.preinstalled()
        deps, preinstalled = _drop_preinstalled(deps, have)
        auto = [a for a in auto if a not in preinstalled]
        # …and what an earlier run in this same conversation already installed.
        deps, cached = _drop_preinstalled(deps, cached_dep_names(deps_cache))
        auto = [a for a in auto if a not in cached]
        try:
            work = Path(tempfile.mkdtemp(prefix="agentexec_", dir=_work_root()))
        except OSError as exc:
            # A missing/unwritable work root (e.g. the AGENT_CODE_EXEC_WORK_ROOT bind mount not
            # present in this deployment) must surface as a TOOL error the agent can report —
            # never crash the whole turn/stream.
            return ExecResult(
                exit_code=None,
                error=(f"code-execution work dir unavailable: {exc}. "
                       f"Check {WORK_ROOT_ENV} and its bind mount in the deployment."),
                backend=self.backend, code=(code or ""),
            )
        try:
            # Continue this conversation's work: bring forward what earlier runs left.
            carried = {}
            if workspace:
                _copy_tree(workspace, work)
                carried = _stat_map(work)
            if entrypoint:
                # The file lives in the workspace and was just carried in; if it is not here,
                # say what IS, so the next call is a correction rather than another guess.
                target = work / entrypoint
                if not target.is_file():
                    present = sorted(
                        str(f.relative_to(work)) for f in work.rglob("*")
                        if f.is_file() and f.relative_to(work).parts[0] not in RESERVED_DIRS)
                    shutil.rmtree(work, ignore_errors=True)
                    return ExecResult(
                        exit_code=None, backend=self.backend, code="",
                        error=(f"no file {entrypoint!r} in the working directory. "
                               f"Files here: {present[:25] or 'none yet'}. "
                               "Write it first with write_workspace_file, or pass the program "
                               "inline as `code`."))
            else:
                # The invariant gate runs on the SAME interpreter, so it inspects the live frames
                # the code produced rather than its source text. Wrapped into the written
                # script but NOT into `code`, so the persisted source stays what the model wrote.
                # PROLOGUE first: it patches imported library units so the contract is enforced.
                script = code or ""
                if invariant_gate_enabled():
                    from agent_runtime.sandbox_verify import epilogue_source, prologue_source
                    script = prologue_source(contracts_for_code(code or "")) + script + epilogue_source()
                (work / "script.py").write_text(script, encoding="utf-8")
            # Stage uploaded/input files into the work dir so the code can read them.
            staged, stage_errors, shadowed = _stage_inputs(work, input_files)
            try:
                os.chmod(work, 0o777)  # let a non-root container user write outputs
            except OSError:
                pass
            from agent_runtime.sandbox_verify import (CHECKS_FILENAME, DECLARED_FILENAME,
                                                      ENVIRONMENT_FILENAME)
            try:
                (work / CHECKS_FILENAME).unlink()
            except OSError:
                pass
            # The tool bridge, when on: a host thread answers the code's capability calls for
            # the length of this run (agent_runtime/tool_bridge.py). Passed only when present,
            # so an executor written against the older _run signature still works.
            bridge = None
            extra_run: Dict[str, Any] = {}
            from agent_runtime import tool_bridge
            if tool_bridge.enabled():
                broot = tool_bridge.bridge_root_for(work)
                bridge = tool_bridge.BridgeServer(
                    broot, container_root=(tool_bridge.CONTAINER_ROOT if self.backend == "docker"
                                           else str(broot)), session=session)
                bridge.start()
                extra_run["bridge_dir"] = broot
            try:
                exit_code, stdout, stderr, timed_out, error = self._run(
                    work, timeout, deps, deps_cache=deps_cache, entrypoint=entrypoint, **extra_run)
            finally:
                if bridge is not None:
                    bridge.stop()
                    bridge.cleanup()
            # Output files the run produced, plus the executed source itself (downloadable).
            # Staged input files are excluded so uploads aren't re-persisted as outputs.
            unchanged = {rel for rel, sig in _stat_map(work).items()
                         if carried.get(rel) == sig}  # carried in and untouched -> not an output
            source_artifacts = _persist_source(code, label=label) if (code or "").strip() else []
            # The gate's own record (environment.json, declared_outputs.json) is provenance, not a
            # result. It stays in the workspace and the run record, beside checks.json, but is not
            # offered as a download: the map UI listed both again after every run.
            artifacts = [*source_artifacts,
                         *_persist_artifacts(work, {"script.py", CHECKS_FILENAME, ENVIRONMENT_FILENAME,
                                                    DECLARED_FILENAME, *staged, *unchanged})]
            verification = _note_printed_outputs(_read_checks(work), stdout, work)
            # The reproducible record: run.py + manifest.json (image DIGEST, in-sandbox
            # environment, input hashes, library slice_shas) + inputs.jsonl. Written into the
            # run dir before it is carried back, so it lands in the durable workspace beside the
            # files it describes; guarded inside emit() so provenance can never fail an analysis.
            # This call existed (M2.8) and the 2026-08-27 merge dropped it without a test noticing.
            if artifacts_enabled():
                from agent_runtime import artifacts as _artifacts
                _artifacts.emit(code=(code or ""), work=work,
                                image=getattr(self, "image", ""), backend=self.backend,
                                dependencies=deps, staged=sorted(staged),
                                verification=verification)
            if workspace:
                _copy_tree(work, workspace, skip={"script.py", CHECKS_FILENAME, *staged})
            if rejected:
                stderr = (str(stderr or "") + f"\n[ignored unsafe dependencies: {rejected}]").strip()
            if auto:
                stderr = (str(stderr or "")
                          + f"\n[installed imports you did not declare: {auto}]").strip()
            size_note = _size_report(artifacts)
            if size_note:
                stderr = (str(stderr or "") + "\n" + size_note).strip()
            if stage_errors:
                stderr = (str(stderr or "") + f"\n[input file staging errors: {stage_errors}]").strip()
            if shadowed:
                stderr = (str(stderr or "") + f"\n[an attached upload was used for {shadowed} "
                          "rather than the file of that name in the working directory; write "
                          "your version under a different name to read it back]").strip()
            # FIRST rather than appended like the notes above: `_clip` keeps the head, and this
            # note is the explanation of the failure, not a footnote to it.
            library_note = _diagnose_library_import(stderr, self.backend)
            if library_note:
                stderr = (library_note + "\n" + str(stderr or "")).strip()
            # Signal-killed runs carry no stderr; surface a cause so the agent can react.
            error = error or _diagnose_abnormal_exit(exit_code, stderr, error)
            return ExecResult(exit_code, _clip(stdout), _clip(stderr), timed_out, error,
                              artifacts, self.backend, code=(code or ""), installed=deps,
                              verification=verification,
                              bridge_calls=(bridge.calls if bridge is not None else []))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _run(self, work: Path, timeout: int,
             dependencies: Optional[List[str]] = None,
             deps_cache: Optional[Path] = None,
             entrypoint: Optional[str] = None,
             bridge_dir: Optional[Path] = None) -> Tuple[Optional[int], str, str, bool, Optional[str]]:
        raise NotImplementedError


class DockerCodeExecutor(CodeExecutor):
    """Container-per-run sandbox (the production backend)."""

    backend = "docker"

    def __init__(self, *, image: str = DEFAULT_IMAGE, memory: str = DEFAULT_MEMORY,
                 cpus: str = DEFAULT_CPUS, pids: str = DEFAULT_PIDS,
                 install_memory: str = DEFAULT_INSTALL_MEMORY) -> None:
        self.image = image
        self.memory = memory
        self.cpus = cpus
        self.pids = pids
        self.install_memory = install_memory

    def preinstalled(self) -> frozenset:
        """Probe the image once per process (results cached per image tag)."""
        override = _preinstalled_override()
        if override is not None:
            return override
        with _probe_lock:
            cached = _probe_cache.get(self.image)
        if cached is not None:
            self._log_image_choice(frozenset(cached))
            return frozenset(cached)
        found = self._probe_versions()
        if found is None:
            # A FAILED probe is deliberately not cached. Caching it would serve the empty set
            # for the life of the process, so one slow first pull (the image not yet local on
            # a Docker-out-of-Docker host) would permanently disable the optimisation with
            # nothing in the logs saying why. An image that genuinely has nothing caches fine
            # — that is an empty frozenset, not None.
            return frozenset()
        with _probe_lock:
            _probe_cache[self.image] = found
        self._log_image_choice(frozenset(found))
        return frozenset(found)

    def preinstalled_versions(self) -> Dict[str, str]:
        """{pip name: version} for what the image ships, or {} when that is unknown."""
        if _preinstalled_override() is not None:
            return {}                      # an explicit list carries no versions to pin to
        self.preinstalled()                # populates the cache
        with _probe_lock:
            return dict(_probe_cache.get(self.image) or {})

    def _probe_versions(self) -> Optional[Dict[str, str]]:
        """Which auto-installable modules the image can already import, or None if unknown.

        Runs under the same no-network/read-only posture and the same resource limits as a
        real run, so the probe can neither widen the sandbox nor become the unbounded
        container that starves it. Returns None — distinct from "the image has nothing" — on
        every failure path, so the caller can decline to cache a failure.
        """
        name = f"agentexec_probe_{uuid.uuid4().hex[:12]}"
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", self.pids,
            "--tmpfs", "/tmp:rw,size=64m", "--env", "HOME=/tmp",
            self.image, "python", "-c", _PROBE_SCRIPT, json.dumps(_IMPORT_TO_PIP),
        ]
        try:
            proc = fork_safe.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        except subprocess.TimeoutExpired:
            # --name exists for exactly this: without the kill the container keeps running
            # after we stop waiting for it. The kill is best-effort — a probe that cannot
            # clean up still has to return "unknown" rather than raise into the caller's run.
            try:
                fork_safe.run(["docker", "kill", name], capture_output=True, timeout=30)
            except Exception:
                pass
            return None
        except (FileNotFoundError, OSError):
            return None
        if proc.returncode != 0:
            return None
        lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
        if not lines:
            return None
        try:
            found = json.loads(lines[-1])
        except ValueError:
            return None
        if not isinstance(found, dict):
            return None
        return {str(k).lower(): str(v or "") for k, v in found.items()}

    def build_argv(self, work: Path, name: str,
                   deps_cache: Optional[Path] = None,
                   entrypoint: Optional[str] = None,
                   bridge_dir: Optional[Path] = None) -> List[str]:
        """Execution phase: NO network, read-only rootfs, deps importable via PYTHONPATH."""
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            # No network during execution, unless the local-only measurement flag says
            # otherwise (exec_network); the install phase below is unchanged either way.
            "--network", exec_network() or "none",
            "--read-only",                  # read-only root fs
            "--cap-drop", "ALL",            # drop all capabilities
            "--security-opt", "no-new-privileges",
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", self.pids,
            "--workdir", "/work",
            "--tmpfs", "/tmp:rw,size=64m,exec",
            "--env", "HOME=/tmp",
            "-v", f"{work}:/work:rw",       # only writable mount
        ]
        # The library mount is part of the extraction bundle: no mount while it is off.
        lib = method_library_dir() if extraction_enabled() else None
        pythonpath = f"/work/{DEPS_DIRNAME}"
        if lib:
            argv += ["-v", f"{lib}:{METHOD_LIBRARY_MOUNT}:ro"]
            pythonpath = f"{METHOD_LIBRARY_MOUNT}:{pythonpath}"
        if bridge_dir is not None:
            # Requests go out through `in`; answers, data and the client module come back
            # through `out`, which the container cannot write (agent_runtime/tool_bridge.py).
            from agent_runtime.tool_bridge import CONTAINER_ROOT
            argv += ["-v", f"{bridge_dir}/in:{CONTAINER_ROOT}/in:rw",
                     "-v", f"{bridge_dir}/out:{CONTAINER_ROOT}/out:ro",
                     "--env", f"IGUIDE_BRIDGE={CONTAINER_ROOT}"]
            pythonpath = f"{pythonpath}:{CONTAINER_ROOT}/out"
        argv += ["--env", f"PYTHONPATH={pythonpath}"]
        if deps_cache is not None:
            # READ-ONLY on purpose. The cache outlives the run, so code that could write to it
            # could leave a poisoned `numpy.py` for the NEXT turn of the conversation to
            # import. Mounting it ro keeps a per-run sandbox escape from becoming a durable one.
            argv += ["-v", f"{deps_cache}:/work/{DEPS_DIRNAME}:ro"]
        user = _host_user()
        if user:
            argv += ["--user", user]
        argv += [self.image, "python", f"/work/{entrypoint or 'script.py'}"]
        return argv

    def build_install_argv(self, work: Path, deps: List[str], name: str,
                           deps_cache: Optional[Path] = None,
                           constraints: Optional[str] = None) -> List[str]:
        """Install phase: network ON, pip install into the shared work dir (/work/.deps)."""
        argv = [
            "docker", "run", "--rm", "--init", "--name", name,
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", self.install_memory,   # larger budget than exec (avoids OOM)
            "--cpus", self.cpus,
            "--pids-limit", self.pids,
            "--workdir", "/work",
            "--tmpfs", "/tmp:rw,size=128m,exec",
            "--env", "HOME=/tmp",
            # pip scratch on the host-backed work dir (real disk), not the tmpfs.
            "--env", f"TMPDIR=/work/{PIPTMP_DIRNAME}",
            "-v", f"{work}:/work:rw",
        ]
        if deps_cache is not None:
            argv += ["-v", f"{deps_cache}:/work/{DEPS_DIRNAME}:rw"]
        user = _host_user()
        if user:
            argv += ["--user", user]
        # --upgrade matters once the target can be pre-populated: pip leaves an existing copy
        # in a --target directory alone and only warns, so without it a cached package would
        # silently pin whatever version got there first and a pinned spec would never apply.
        argv += [self.image, "pip", "install", "--no-cache-dir", "--upgrade",
                 "--target", f"/work/{DEPS_DIRNAME}"]
        if constraints:
            # Lives under .piptmp because that directory is already excluded from artifacts
            # and from the workspace copy-out — a constraints file is machinery, not output.
            argv += ["--constraint", f"/work/{PIPTMP_DIRNAME}/{constraints}"]
        argv += [*deps]
        return argv

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             deps_cache: Optional[Path] = None, entrypoint: Optional[str] = None,
             bridge_dir: Optional[Path] = None):
        # Phase 1 (deps): a separate container WITH network installs into /work/.deps.
        # Nothing to install is the common case once the image is baked and the cache is warm,
        # and then this phase — a whole container start — is skipped entirely.
        if dependencies:
            piptmp = work / PIPTMP_DIRNAME
            piptmp.mkdir(parents=True, exist_ok=True)  # pip TMPDIR (real disk)
            constraints_name = None
            text = _constraints_text(self.preinstalled_versions())
            if text:
                constraints_name = "image-constraints.txt"
                try:
                    (piptmp / constraints_name).write_text(text, encoding="utf-8")
                except OSError:
                    constraints_name = None
            iname = f"agentexec_pip_{uuid.uuid4().hex[:12]}"
            try:
                inst = fork_safe.run(
                    self.build_install_argv(work, dependencies, iname, deps_cache, constraints_name),
                    capture_output=True, text=True, timeout=DEFAULT_INSTALL_TIMEOUT + 5)
            except subprocess.TimeoutExpired as exc:
                fork_safe.run(["docker", "kill", iname], capture_output=True)
                _evict_torn_cache(deps_cache)
                return None, (exc.stdout or ""), (exc.stderr or ""), True, "dependency install timed out"
            except FileNotFoundError:
                return None, "", "", False, "docker executable not found"
            if inst.returncode != 0:
                _evict_torn_cache(deps_cache)
                return None, inst.stdout, _clip(inst.stderr), False, "dependency install failed"
        # Phase 2 (exec): NO network.
        name = f"agentexec_{uuid.uuid4().hex[:12]}"
        try:
            proc = fork_safe.run(self.build_argv(work, name, deps_cache, entrypoint,
                                                 **({"bridge_dir": bridge_dir} if bridge_dir else {})),
                                 capture_output=True, text=True, timeout=timeout + 5)
            return proc.returncode, proc.stdout, proc.stderr, False, None
        except subprocess.TimeoutExpired as exc:
            fork_safe.run(["docker", "kill", name], capture_output=True)
            return None, (exc.stdout or ""), (exc.stderr or ""), True, "execution timed out"
        except FileNotFoundError:
            return None, "", "", False, "docker executable not found"
        except Exception as exc:  # pragma: no cover - defensive
            return None, "", "", False, f"{type(exc).__name__}: {exc}"


class LocalSubprocessExecutor(CodeExecutor):
    """DEV ONLY — runs on the host, NOT a sandbox. Opt-in via AGENT_CODE_EXEC_BACKEND=local."""

    backend = "local-unsafe"

    def preinstalled(self) -> frozenset:
        """The host interpreter's own packages — no container to probe, so ask it directly."""
        override = _preinstalled_override()
        if override is not None:
            return override
        return frozenset(self.preinstalled_versions())

    def preinstalled_versions(self) -> Dict[str, str]:
        override = _preinstalled_override()
        if override is not None:
            return {}
        import importlib
        from importlib.metadata import version as _dist_version

        found: Dict[str, str] = {}
        for mod, pip in _IMPORT_TO_PIP.items():
            try:
                importlib.import_module(mod)     # a real import, for the reason above
            except Exception:
                continue
            try:
                found[pip.lower()] = _dist_version(pip)
            except Exception:
                found[pip.lower()] = ""
        return found

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             deps_cache: Optional[Path] = None, entrypoint: Optional[str] = None,
             bridge_dir: Optional[Path] = None):
        # No container to mount into, so the durable cache IS the target directory.
        deps_dir = deps_cache if deps_cache is not None else (work / DEPS_DIRNAME)
        if dependencies:
            try:
                inst = fork_safe.run(
                    [sys.executable or "python", "-m", "pip", "install", "--no-cache-dir",
                     "--upgrade", "--target", str(deps_dir), *dependencies],
                    capture_output=True, text=True, timeout=DEFAULT_INSTALL_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                return None, (exc.stdout or ""), (exc.stderr or ""), True, "dependency install timed out"
            if inst.returncode != 0:
                return None, inst.stdout, _clip(inst.stderr), False, "dependency install failed"
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(work)}
        # Also when this run installed nothing: an earlier run in the conversation may have.
        paths = []
        lib = method_library_dir() if extraction_enabled() else None
        if lib:
            paths.append(str(lib))
        if dependencies or deps_cache is not None:
            paths.append(str(deps_dir))
        if bridge_dir is not None:
            env["IGUIDE_BRIDGE"] = str(bridge_dir)
            paths.append(str(Path(bridge_dir) / "out"))
        if paths:
            env["PYTHONPATH"] = os.pathsep.join(paths)
        try:
            proc = fork_safe.run(
                [sys.executable or "python", entrypoint or "script.py"],
                cwd=str(work), env=env, capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, proc.stdout, proc.stderr, False, None
        except subprocess.TimeoutExpired as exc:
            return None, (exc.stdout or ""), (exc.stderr or ""), True, "execution timed out"
        except Exception as exc:  # pragma: no cover - defensive
            return None, "", "", False, f"{type(exc).__name__}: {exc}"


class DisabledExecutor(CodeExecutor):
    """Returned when the requested backend is unavailable; never runs code."""

    backend = "disabled"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def _run(self, work: Path, timeout: int, dependencies: Optional[List[str]] = None,
             deps_cache: Optional[Path] = None, entrypoint: Optional[str] = None,
             bridge_dir: Optional[Path] = None):
        return None, "", "", False, self._reason


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def get_code_executor() -> CodeExecutor:
    """Pick an executor from env.

    ``AGENT_CODE_EXEC_BACKEND``: ``docker`` (default) | ``local`` (UNSAFE, dev).
    Docker is *not* silently downgraded to local — if docker is unavailable a
    DisabledExecutor is returned so untrusted code is never run unsandboxed.
    """
    backend = (os.getenv("AGENT_CODE_EXEC_BACKEND") or "docker").strip().lower()
    if backend == "local":
        return LocalSubprocessExecutor()
    if backend == "docker":
        if _docker_available():
            return DockerCodeExecutor()
        return DisabledExecutor(
            "docker backend unavailable (docker not found). Set AGENT_CODE_EXEC_BACKEND=local "
            "to run on the host for development (UNSAFE — not a sandbox)."
        )
    return DisabledExecutor(f"unknown AGENT_CODE_EXEC_BACKEND: {backend}")


__all__ = [
    "ExecResult",
    "CodeExecutor",
    "DockerCodeExecutor",
    "LocalSubprocessExecutor",
    "DisabledExecutor",
    "get_code_executor",
    "is_code_exec_enabled",
]
