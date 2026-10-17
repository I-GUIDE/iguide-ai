"""JSON that a browser can parse: non-finite numbers written as null.

Python's ``json`` writes ``float('nan')`` as the bare token ``NaN`` (and infinities as
``Infinity``), which no strict JSON parser accepts. A browser's ``JSON.parse`` throws on it. On
2026-10-17 (round r59 of the GIS assessment log) a route GeoJSON written by execute_code carried
pandas NaN for five unnamed ways in a ``streets`` list. add_map_layer re-wrote the file through
GDAL, which nulls a non-finite SCALAR but copies a list property through as raw JSON, so the NaN
survived. The file was served with a 200, the map's JSON.parse threw "Unexpected token 'N'", and
the answer told the user the route was on the map.

Two entry points, for the two ways such a value reaches a browser:

* :func:`strict_json_bytes` for a FILE. The download route runs every ``.geojson`` it serves
  through it, so the fix covers every layer producer at once: add_map_layer, a code peer's raw
  outputs (map_layers.layers_for_artifacts points the map straight at them), any tool's file,
  and the files already in the store.
* :func:`strict_dumps` for an OBJECT about to be serialized, such as an SSE event that carries
  a layer's GeoJSON inline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple

# A file without either token cannot hold a non-finite value Python's json would accept, so
# most files are answered by a substring test and never parsed. ("-Infinity" contains
# "Infinity".) A number too large for a float, such as 1e999, is valid JSON that JSON.parse reads
# as Infinity without complaint, so it does not need the parse either.
_NON_FINITE_TOKENS = (b"NaN", b"Infinity")


def strict_json_bytes(raw: bytes) -> Tuple[Optional[bytes], int]:
    """``(strict bytes, count)`` for JSON holding NaN/Infinity, or ``(None, 0)`` if none.

    ``None`` means "serve the original unchanged": the text holds no non-finite value, or it is
    not JSON at all (the client then reports a file it cannot read, which is the truth). The
    values are replaced wherever they sit, in a scalar property, inside a list, or inside a
    nested object, because the parser sees them all.
    """
    if not any(token in raw for token in _NON_FINITE_TOKENS):
        return None, 0
    replaced = 0

    def _constant(_name: str) -> None:
        nonlocal replaced
        replaced += 1
        return None

    def _float(text: str) -> Optional[float]:
        nonlocal replaced
        value = float(text)
        if math.isfinite(value):
            return value
        replaced += 1
        return None

    try:
        data = json.loads(raw, parse_constant=_constant, parse_float=_float)
    except ValueError:
        return None, 0
    if not replaced:
        return None, 0
    out = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    return out.encode("utf-8"), replaced


def sanitize_json_file(path: Any) -> int:
    """Rewrite ``path`` as strict JSON in place; the number of values written as null."""
    target = Path(str(path))
    fixed, count = strict_json_bytes(target.read_bytes())
    if fixed is not None:
        target.write_bytes(fixed)
    return count


def _nulled(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _nulled(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nulled(v) for v in value]
    return value


def strict_dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` that writes NaN/Infinity as null instead of an invalid token.

    The first attempt is the plain strict dump, so an object with no non-finite float costs
    nothing extra. Only one that fails is walked and dumped again.
    """
    kwargs.pop("allow_nan", None)
    try:
        return json.dumps(obj, allow_nan=False, **kwargs)
    except ValueError:
        return json.dumps(_nulled(obj), allow_nan=False, **kwargs)
