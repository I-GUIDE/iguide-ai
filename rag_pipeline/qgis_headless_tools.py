from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from .agent_file_store import create_output_file_from_path, get_file_record, resolve_file_id, storage_root


DEFAULT_QGIS_PROCESS_BIN = "qgis_process"
DEFAULT_QGIS_PYTHON_BIN = sys.executable


def _qgis_force_override() -> Optional[bool]:
    """``AGENT_QGIS_ENABLED`` as a tri-state: True/False to force, None to auto-detect."""
    raw = os.getenv("AGENT_QGIS_ENABLED")
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def qgis_process_bin() -> Optional[str]:
    """Resolved ``qgis_process`` executable, or None when QGIS is genuinely absent.

    ``QGIS_PROCESS_BIN`` is honored when it resolves, but a configured path that does NOT exist
    here falls back to the plain name on PATH — otherwise a developer .env pointing at a local
    QGIS install (e.g. a macOS /Applications path) silently disables QGIS inside the Linux
    container, where the binary is on PATH all along.
    """
    configured = (os.getenv("QGIS_PROCESS_BIN") or "").strip()
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
    return shutil.which(DEFAULT_QGIS_PROCESS_BIN)


def qgis_process_available() -> bool:
    """Whether the ``qgis_process`` CLI (used by the processing/buffer tools) is usable."""
    forced = _qgis_force_override()
    if forced is not None:
        return forced
    return qgis_process_bin() is not None


_PYQGIS_PROBE_CACHE: Dict[str, bool] = {}


def qgis_python_candidates() -> List[str]:
    """Interpreters to probe for the PyQGIS bindings, in priority order.

    ``QGIS_PYTHON_BIN`` first (the image sets it to the distro python that ``python3-qgis``
    targets), then this interpreter, then the distro python — so a configured path that does
    not exist on this host (e.g. a dev machine's QGIS.app python leaking in through .env) does
    not disable PyQGIS when a working interpreter is available. Each candidate is only accepted
    if it can actually import ``qgis``.
    """
    candidates: List[str] = []
    configured = (os.getenv("QGIS_PYTHON_BIN") or "").strip()
    for candidate in (configured, sys.executable, "/usr/bin/python3"):
        if not candidate or candidate in candidates:
            continue
        if candidate == sys.executable or os.path.exists(candidate) or shutil.which(candidate):
            candidates.append(candidate)
    return candidates


def pyqgis_available() -> bool:
    """Whether PyQGIS (render_map / layer_summary) is importable by the WORKER interpreter.

    The PyQGIS bindings live in the interpreter that runs ``qgis_pyqgis_worker.py`` —
    ``QGIS_PYTHON_BIN`` — which in the Docker image is the distro ``python3`` that
    ``apt install python3-qgis`` targets, NOT the app's interpreter. So we probe THAT
    interpreter for the ``qgis`` module (cached per interpreter), rather than the current one.
    """
    forced = _qgis_force_override()
    if forced is not None:
        return forced
    for python_bin in qgis_python_candidates():
        if python_bin in _PYQGIS_PROBE_CACHE:
            if _PYQGIS_PROBE_CACHE[python_bin]:
                return True
            continue
        ok = False
        try:
            if python_bin == sys.executable:
                ok = importlib.util.find_spec("qgis") is not None
            else:
                probe = subprocess.run(
                    [python_bin, "-c", "import importlib.util as u, sys; sys.exit(0 if u.find_spec('qgis') else 1)"],
                    capture_output=True, timeout=15,
                )
                ok = probe.returncode == 0
        except Exception:
            ok = False
        _PYQGIS_PROBE_CACHE[python_bin] = ok
        if ok:
            return True
    return False


def qgis_python_bin() -> Optional[str]:
    """The first candidate interpreter that can import ``qgis``, or None.

    Mirrors :func:`pyqgis_available` so DETECTION and EXECUTION always agree on the interpreter
    (a configured path that does not exist here must not be used to launch the worker).
    """
    for candidate in qgis_python_candidates():
        if candidate in _PYQGIS_PROBE_CACHE:
            if _PYQGIS_PROBE_CACHE[candidate]:
                return candidate
            continue
        if pyqgis_available():          # populates the cache for the candidates in order
            for cached, ok in _PYQGIS_PROBE_CACHE.items():
                if ok:
                    return cached
        return None
    return None


def qgis_available() -> bool:
    """Whether ANY QGIS backend (CLI or PyQGIS) is usable.

    The agent image ships GDAL (for the geopandas-backed geo tools) but NOT QGIS, so this is
    typically False — callers then expose the geopandas tools (``render_map_image`` etc.) instead
    of QGIS tools that would only fail at call time. Forceable via ``AGENT_QGIS_ENABLED``.
    """
    return qgis_process_available() or pyqgis_available()
DEFAULT_PROCESSING_TIMEOUT_SEC = 300
DEFAULT_PYQGIS_TIMEOUT_SEC = 180
OUTPUT_PARAMETER_HINTS = {
    "DESTINATION",
    "OUTPUT",
    "OUTPUT_FILE",
    "OUTPUT_HTML_FILE",
    "OUTPUT_LAYER",
    "OUTPUT_TABLE",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _safe_session_id(session_id: Optional[str]) -> str:
    value = (session_id or "default").strip() or "default"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120] or "default"


def _qgis_jobs_root() -> Path:
    configured = os.getenv("QGIS_JOB_ROOT")
    root = Path(configured).expanduser().resolve() if configured else storage_root() / "qgis_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_job_dir(session_id: Optional[str]) -> tuple[str, Path]:
    job_id = f"qgis_{uuid4().hex[:12]}"
    job_dir = _qgis_jobs_root() / _safe_session_id(session_id) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def _json_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)


def _parse_json_object(raw: str, *, field_name: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object string: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object string.")
    return dict(parsed)


def _parse_json_array(raw: str, *, field_name: str) -> list[Any]:
    try:
        parsed = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON array string: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON array string.")
    return list(parsed)


# Components of an (extracted) ESRI shapefile set — any one of these can reference the layer.
_SHAPE_PARTS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qpj", ".aih", ".ain"}


def _orig_name(path: Path, record: Optional[Mapping[str, Any]]) -> str:
    """The original filename (uploads are stored on disk as ``<file_id>__<filename>``)."""
    if record and record.get("filename"):
        return str(record["filename"])
    name = path.name
    return name.split("__", 1)[1] if "__" in name else name


def _stage_shapefile_siblings(part_path: Path, record: Optional[Mapping[str, Any]]) -> str:
    """Co-locate an uploaded shapefile's parts so OGR/QGIS can open it.

    Uploads are stored as ``uploads/<file_id>__<filename>`` — so a ``.shp`` and its
    ``.shx``/``.dbf``/``.prj`` siblings are NOT next to each other under a shared basename and
    GDAL can't find them. Given ANY one component, gather every upload sharing the same basename
    stem, copy them into one temp dir under a common name, and return the staged ``.shp`` path.
    Falls back to the original path when no ``.shp`` is present (OGR then fails clearly).
    """
    name = _orig_name(part_path, record)
    stem = Path(name).stem
    members: Dict[str, Path] = {}
    try:
        for sibling in part_path.parent.iterdir():
            if not sibling.is_file():
                continue
            on = sibling.name.split("__", 1)[1] if "__" in sibling.name else sibling.name
            ext = Path(on).suffix.lower()
            if Path(on).stem == stem and ext in _SHAPE_PARTS:
                members.setdefault(ext, sibling)
    except OSError:
        pass
    members.setdefault(Path(name).suffix.lower(), part_path)
    if ".shp" not in members:
        return str(part_path)
    tmp = Path(tempfile.mkdtemp(prefix="qgis_shp_"))
    for ext, src in members.items():
        try:
            shutil.copyfile(src, tmp / f"{stem}{ext}")
        except OSError:
            pass
    return str(tmp / f"{stem}.shp")


def _resolve_layer_ref(ref: Any) -> Any:
    if not isinstance(ref, str):
        return ref
    value = ref.strip()
    if not value:
        return value
    record = get_file_record(value)
    if record:
        path = Path(resolve_file_id(value))
        ext = Path(_orig_name(path, record)).suffix.lower()
        if ext == ".zip":  # GDAL reads the zipped shapefile in place
            return f"/vsizip/{path.resolve()}"
        if ext in _SHAPE_PARTS:  # extracted shapefile: re-assemble the sibling parts
            return _stage_shapefile_siblings(path, record)
        return str(path)
    return value


def _normalize_layer_specs(layers: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for item in layers:
        if isinstance(item, str):
            normalized.append({"path": _resolve_layer_ref(item), "provider": "ogr"})
            continue
        if isinstance(item, dict):
            copy = dict(item)
            for key in ("path", "layer_path", "INPUT", "input"):
                if key in copy:
                    copy[key] = _resolve_layer_ref(copy[key])
            normalized.append(copy)
            continue
        normalized.append(item)
    return normalized


def _bounded_timeout(value: Any, *, default: int, maximum: int = 1800) -> int:
    return _bounded_int(value, default=default, minimum=1, maximum=maximum)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _looks_like_output_parameter(name: str) -> bool:
    upper = name.upper()
    return upper in OUTPUT_PARAMETER_HINTS or upper.startswith("OUTPUT_")


def _normalize_processing_parameters(parameters: Mapping[str, Any], job_dir: Path) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in parameters.items():
        if _looks_like_output_parameter(str(key)) and isinstance(value, str):
            resolved_value = value
        else:
            resolved_value = _resolve_layer_ref(value)
        if (
            _looks_like_output_parameter(str(key))
            and isinstance(resolved_value, str)
            and resolved_value
            and resolved_value.upper() not in {"TEMPORARY_OUTPUT", "TEMPORARY_OUTPUTS"}
            and not Path(resolved_value).expanduser().is_absolute()
        ):
            normalized[str(key)] = str(job_dir / resolved_value)
        else:
            normalized[str(key)] = resolved_value
    return normalized


def _parameter_value_to_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return ""
    return str(value)


def _qgis_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    repo_root = str(_repo_root())
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else repo_root
    return env


def _run_subprocess(command: list[str], *, timeout_sec: int, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd or _repo_root()),
        env=_qgis_env(),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def _parse_stdout_json(stdout: str) -> Any:
    if not stdout.strip():
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def qgis_processing_help_tool(algorithm: str, timeout_sec: int = 60) -> str:
    """Return qgis_process JSON help for a Processing algorithm id."""
    algorithm = str(algorithm or "").strip()
    if not algorithm:
        raise ValueError("algorithm is required, for example 'native:buffer'.")

    # Resolved (config first, PATH fallback) so a dev .env path that does not exist here
    # cannot break execution while detection reports QGIS available.
    qgis_process = qgis_process_bin() or os.getenv("QGIS_PROCESS_BIN", DEFAULT_QGIS_PROCESS_BIN)
    timeout = _bounded_timeout(timeout_sec, default=60, maximum=300)
    command = [qgis_process, "--json", "help", algorithm]

    try:
        completed = _run_subprocess(command, timeout_sec=timeout)
    except FileNotFoundError as exc:
        return _json_payload(
            {
                "ok": False,
                "error": f"qgis_process not found: {qgis_process}. Set QGIS_PROCESS_BIN.",
                "exception": str(exc),
            }
        )
    except subprocess.TimeoutExpired as exc:
        return _json_payload(
            {
                "ok": False,
                "error": "qgis_process help timed out",
                "timeout_sec": timeout,
                "exception": str(exc),
            }
        )

    return _json_payload(
        {
            "ok": completed.returncode == 0,
            "algorithm": algorithm,
            "command": command,
            "returncode": completed.returncode,
            "stdout_json": _parse_stdout_json(completed.stdout),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )


def qgis_processing_run_tool(
    algorithm: str,
    parameters_json: str,
    session_id: Optional[str] = None,
    timeout_sec: int = DEFAULT_PROCESSING_TIMEOUT_SEC,
) -> str:
    """Run one QGIS Processing algorithm in a headless subprocess."""
    algorithm = str(algorithm or "").strip()
    if not algorithm:
        raise ValueError("algorithm is required, for example 'native:buffer'.")

    parameters = _parse_json_object(parameters_json, field_name="parameters_json")
    timeout = _bounded_timeout(timeout_sec, default=DEFAULT_PROCESSING_TIMEOUT_SEC)
    job_id, job_dir = _new_job_dir(session_id)
    effective_parameters = _normalize_processing_parameters(parameters, job_dir)
    (job_dir / "parameters.json").write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "input_parameters": parameters,
                "effective_parameters": effective_parameters,
            },
            ensure_ascii=True,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Resolved (config first, PATH fallback) so a dev .env path that does not exist here
    # cannot break execution while detection reports QGIS available.
    qgis_process = qgis_process_bin() or os.getenv("QGIS_PROCESS_BIN", DEFAULT_QGIS_PROCESS_BIN)
    cli_parameters = [f"{key}={_parameter_value_to_cli(value)}" for key, value in effective_parameters.items()]
    command = [qgis_process, "--json", "run", algorithm, "--", *cli_parameters]

    try:
        completed = _run_subprocess(command, timeout_sec=timeout, cwd=job_dir)
    except FileNotFoundError as exc:
        return _json_payload(
            {
                "ok": False,
                "job_id": job_id,
                "job_dir": str(job_dir),
                "error": f"qgis_process not found: {qgis_process}. Set QGIS_PROCESS_BIN.",
                "exception": str(exc),
            }
        )
    except subprocess.TimeoutExpired as exc:
        return _json_payload(
            {
                "ok": False,
                "job_id": job_id,
                "job_dir": str(job_dir),
                "algorithm": algorithm,
                "error": "qgis_process run timed out",
                "timeout_sec": timeout,
                "exception": str(exc),
            }
        )

    (job_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (job_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    return _json_payload(
        {
            "ok": completed.returncode == 0,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "algorithm": algorithm,
            "effective_parameters": effective_parameters,
            "command": command,
            "returncode": completed.returncode,
            "stdout_json": _parse_stdout_json(completed.stdout),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )


def qgis_metric_buffer_tool(
    input_layer: str,
    distance_meters: float,
    output_filename: str = "buffer.geojson",
    projected_crs: str = "EPSG:26916",
    target_crs: str = "EPSG:4326",
    dissolve: bool = False,
    segments: int = 12,
    session_id: Optional[str] = None,
    timeout_sec: int = DEFAULT_PROCESSING_TIMEOUT_SEC,
) -> str:
    """Create a metric buffer by reprojecting before running QGIS native:buffer."""
    source = str(_resolve_layer_ref(str(input_layer or "").strip()))
    if not source:
        raise ValueError("input_layer is required.")

    try:
        distance = float(distance_meters)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_meters must be a number.") from exc
    if distance <= 0:
        raise ValueError("distance_meters must be greater than zero.")

    timeout = _bounded_timeout(timeout_sec, default=DEFAULT_PROCESSING_TIMEOUT_SEC)
    segment_count = _bounded_int(segments, default=12, minimum=1, maximum=96)
    job_id, job_dir = _new_job_dir(session_id)
    # Resolved (config first, PATH fallback) so a dev .env path that does not exist here
    # cannot break execution while detection reports QGIS available.
    qgis_process = qgis_process_bin() or os.getenv("QGIS_PROCESS_BIN", DEFAULT_QGIS_PROCESS_BIN)
    safe_output_name = Path(output_filename or "buffer.geojson").name
    projected_input = job_dir / "input_projected.gpkg"
    buffered_projected = job_dir / "buffer_projected.gpkg"
    final_output = job_dir / safe_output_name

    steps = [
        {
            "name": "reproject_input",
            "algorithm": "native:reprojectlayer",
            "parameters": {
                "INPUT": source,
                "TARGET_CRS": projected_crs,
                "OUTPUT": str(projected_input),
            },
        },
        {
            "name": "buffer_projected",
            "algorithm": "native:buffer",
            "parameters": {
                "INPUT": str(projected_input),
                "DISTANCE": distance,
                "SEGMENTS": segment_count,
                "DISSOLVE": bool(dissolve),
                "OUTPUT": str(buffered_projected),
            },
        },
        {
            "name": "reproject_output",
            "algorithm": "native:reprojectlayer",
            "parameters": {
                "INPUT": str(buffered_projected),
                "TARGET_CRS": target_crs,
                "OUTPUT": str(final_output),
            },
        },
    ]
    (job_dir / "metric_buffer_parameters.json").write_text(
        json.dumps(
            {
                "input_layer": input_layer,
                "resolved_input_layer": source,
                "distance_meters": distance,
                "projected_crs": projected_crs,
                "target_crs": target_crs,
                "dissolve": bool(dissolve),
                "segments": segment_count,
                "output_path": str(final_output),
            },
            ensure_ascii=True,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    executed_steps = []
    for step in steps:
        command = [
            qgis_process,
            "--json",
            "run",
            str(step["algorithm"]),
            "--",
            *[
                f"{key}={_parameter_value_to_cli(value)}"
                for key, value in (step.get("parameters") or {}).items()
            ],
        ]
        try:
            completed = _run_subprocess(command, timeout_sec=timeout, cwd=job_dir)
        except FileNotFoundError as exc:
            return _json_payload(
                {
                    "ok": False,
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "error": f"qgis_process not found: {qgis_process}. Set QGIS_PROCESS_BIN.",
                    "exception": str(exc),
                    "failed_step": step["name"],
                }
            )
        except subprocess.TimeoutExpired as exc:
            return _json_payload(
                {
                    "ok": False,
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "error": "metric buffer step timed out",
                    "timeout_sec": timeout,
                    "exception": str(exc),
                    "failed_step": step["name"],
                }
            )

        step_payload = {
            "name": step["name"],
            "algorithm": step["algorithm"],
            "parameters": step["parameters"],
            "command": command,
            "returncode": completed.returncode,
            "stdout_json": _parse_stdout_json(completed.stdout),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        executed_steps.append(step_payload)
        (job_dir / f"{step['name']}_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (job_dir / f"{step['name']}_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            return _json_payload(
                {
                    "ok": False,
                    "job_id": job_id,
                    "job_dir": str(job_dir),
                    "failed_step": step["name"],
                    "steps": executed_steps,
                }
            )

    payload: Dict[str, Any] = {
        "ok": final_output.exists(),
        "job_id": job_id,
        "job_dir": str(job_dir),
        "input_layer": source,
        "distance_meters": distance,
        "projected_crs": projected_crs,
        "target_crs": target_crs,
        "output_path": str(final_output),
        "steps": executed_steps,
    }
    if final_output.exists():
        try:
            payload["managed_output"] = create_output_file_from_path(
                final_output,
                filename=final_output.name,
                overwrite=True,
            )
        except Exception as exc:
            payload["managed_output_error"] = str(exc)
    return _json_payload(payload)


def _run_pyqgis_worker(
    operation: str,
    spec: Mapping[str, Any],
    *,
    session_id: Optional[str],
    timeout_sec: int,
) -> Dict[str, Any]:
    job_id, job_dir = _new_job_dir(session_id)
    spec_path = job_dir / "job_spec.json"
    result_path = job_dir / "result.json"
    payload = dict(spec)
    payload["job_id"] = job_id
    payload["job_dir"] = str(job_dir)
    payload["result_path"] = str(result_path)
    spec_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")

    # First candidate interpreter that can actually import qgis (config, this python, distro).
    python_bin = qgis_python_bin() or os.getenv("QGIS_PYTHON_BIN", DEFAULT_QGIS_PYTHON_BIN)
    worker_path = _repo_root() / "rag_pipeline" / "qgis_pyqgis_worker.py"
    command = [python_bin, str(worker_path), operation, str(spec_path)]
    try:
        completed = _run_subprocess(command, timeout_sec=timeout_sec, cwd=_repo_root())
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "error": f"QGIS Python executable not found: {python_bin}. Set QGIS_PYTHON_BIN.",
            "exception": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "job_id": job_id,
            "job_dir": str(job_dir),
            "error": "PyQGIS worker timed out",
            "timeout_sec": timeout_sec,
            "exception": str(exc),
        }

    (job_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (job_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    result: Dict[str, Any] = {}
    if result_path.exists():
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                result = parsed
        except Exception as exc:
            result = {"ok": False, "error": f"could not parse PyQGIS result: {exc}"}
    if not result:
        result = {"ok": completed.returncode == 0}

    result.update(
        {
            "job_id": job_id,
            "job_dir": str(job_dir),
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.returncode != 0 and result.get("ok") is not False:
        result["ok"] = False
    return result


def pyqgis_layer_summary_tool(
    layer_path: str,
    provider: str = "ogr",
    layer_name: Optional[str] = None,
    sample_limit: int = 5,
    session_id: Optional[str] = None,
    timeout_sec: int = DEFAULT_PYQGIS_TIMEOUT_SEC,
) -> str:
    """Inspect one vector or raster layer with a standalone headless PyQGIS process."""
    path = str(layer_path or "").strip()
    if not path:
        raise ValueError("layer_path is required.")
    path = str(_resolve_layer_ref(path))
    timeout = _bounded_timeout(timeout_sec, default=DEFAULT_PYQGIS_TIMEOUT_SEC)
    spec = {
        "layer_path": path,
        "provider": str(provider or "ogr"),
        "layer_name": layer_name,
        "sample_limit": _bounded_int(sample_limit, default=5, minimum=0, maximum=50),
    }
    return _json_payload(_run_pyqgis_worker("layer_summary", spec, session_id=session_id, timeout_sec=timeout))


def pyqgis_render_map_tool(
    layers_json: str,
    output_filename: str = "map.png",
    width: int = 1200,
    height: int = 800,
    extent_json: Optional[str] = None,
    basemap: str = "none",
    basemap_url: Optional[str] = None,
    crs: Optional[str] = None,
    session_id: Optional[str] = None,
    timeout_sec: int = DEFAULT_PYQGIS_TIMEOUT_SEC,
) -> str:
    """Render vector/raster layers to a PNG with a standalone headless PyQGIS process."""
    layers = _normalize_layer_specs(_parse_json_array(layers_json, field_name="layers_json"))
    if not layers:
        raise ValueError("layers_json must contain at least one layer object.")
    timeout = _bounded_timeout(timeout_sec, default=DEFAULT_PYQGIS_TIMEOUT_SEC)
    spec = {
        "layers": layers,
        "output_filename": output_filename or "map.png",
        "width": _bounded_int(width, default=1200, minimum=1, maximum=8000),
        "height": _bounded_int(height, default=800, minimum=1, maximum=8000),
        "extent_json": extent_json,
        "basemap": basemap,
        "basemap_url": basemap_url,
        "crs": crs,
    }
    result = _run_pyqgis_worker("render_map", spec, session_id=session_id, timeout_sec=timeout)
    output_path = result.get("output_path")
    if result.get("ok") and output_path:
        try:
            record = create_output_file_from_path(
                output_path,
                filename=Path(str(output_path)).name,
                overwrite=True,
            )
            result["managed_output"] = record
        except Exception as exc:
            result["managed_output_error"] = str(exc)
    return _json_payload(result)


__all__ = [
    "pyqgis_layer_summary_tool",
    "pyqgis_render_map_tool",
    "qgis_metric_buffer_tool",
    "qgis_processing_help_tool",
    "qgis_processing_run_tool",
]
