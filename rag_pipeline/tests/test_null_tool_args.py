"""An explicit `null` means "use the default", on every optional tool parameter.

A model writing tool arguments fills EVERY slot the schema offers and puts null in the ones it
does not need. Where a parameter was annotated `str` with a default, pydantic refused the call
before the tool ran. Measured live on gpt-oss:120b:

    embed_region({..., 'buffer_m': None, 'lon': None, 'lat': None, 'bbox': None,
                  'start': None, 'end': None})
    ValidationError: 3 validation errors for embed_region

Nothing was wrong with that call. start=None means "no opinion about the date window", which is
precisely what the default expresses — the tool had a good answer and refused to use it over a
type annotation. A scan found 141 such parameters across 62 of 80 exposed tools.

It cannot be fixed inside the function: pydantic validates against the SCHEMA, inferred from the
signature, so the call dies before any of our code runs. The signature has to say "nullable",
which is what the wrapper rewrites.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest
from langchain_core.tools import StructuredTool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_runtime.tool_args import accept_null_defaults  # noqa: E402


def sample(bbox: Optional[List[float]] = None, start: str = "2022-06",
           buffer_m: float = 2000.0, count: int = 3, flag: bool = True,
           required: str = None) -> str:  # noqa: RUF013 - deliberate: default-None stays as-is
    return f"{start}|{buffer_m}|{count}|{flag}|{bbox}|{required}"


def tool_for(func):
    return StructuredTool.from_function(func=func, name="sample", description="d")


# --- the live failure ------------------------------------------------------------

def test_the_call_that_failed_live_now_works():
    out = tool_for(accept_null_defaults(sample)).invoke(
        {"bbox": None, "start": None, "buffer_m": None, "count": None, "flag": None})
    assert out == "2022-06|2000.0|3|True|None|None"


def test_it_really_was_rejected_before():
    """Guards the premise. If this ever stops raising, the wrapper is solving nothing."""
    with pytest.raises(Exception) as exc:
        tool_for(sample).invoke({"start": None, "buffer_m": None})
    assert "validation" in str(exc.value).lower()


# --- and nothing else changes ----------------------------------------------------

def test_a_real_value_still_wins():
    out = tool_for(accept_null_defaults(sample)).invoke({"start": "2023-01", "count": 9})
    assert out.startswith("2023-01|2000.0|9|")


def test_omitting_a_parameter_is_unchanged():
    assert tool_for(accept_null_defaults(sample)).invoke({}) == "2022-06|2000.0|3|True|None|None"


def test_falsy_values_are_not_treated_as_null():
    """0 and False are ANSWERS. Substituting the default for them would silently ignore the
    caller — a far worse bug than the one being fixed."""
    out = tool_for(accept_null_defaults(sample)).invoke({"count": 0, "flag": False})
    assert "|0|False|" in out


def test_a_required_parameter_stays_required():
    def needs_one(a: str, b: int = 2) -> str:
        return f"{a}{b}"
    with pytest.raises(Exception):
        tool_for(accept_null_defaults(needs_one)).invoke({"b": 5})


def test_a_function_with_no_defaults_is_returned_untouched():
    def plain(a: str) -> str:
        return a
    assert accept_null_defaults(plain) is plain


# --- across the real tool surface -------------------------------------------------

def _all_tools():
    from agent_runtime.rs_embed_tools import make_rs_embed_tools
    from agent_runtime.terrain_tools import make_terrain_tools
    from agent_runtime.analysis_overlay_tools import make_overlay_tools
    from agent_runtime.analysis_aggregate_tools import make_aggregate_tools
    from agent_runtime.admin_boundary_tools import make_admin_boundary_tools
    out = []
    for make in (make_rs_embed_tools, make_terrain_tools, make_overlay_tools,
                 make_aggregate_tools, make_admin_boundary_tools):
        try:
            out.extend(make())
        except Exception:  # noqa: BLE001 - a toolset needing credentials is not this test's job
            continue
    return out


def test_no_exposed_tool_still_refuses_a_null_optional():
    """The scan that prompted this found 141 offenders. The point is that there are now none."""
    offenders = []
    for tool in _all_tools():
        schema = tool.args_schema
        fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", {})
        for fname, field in fields.items():
            required = getattr(field, "is_required", None)
            if callable(required):
                required = required()
            if required:
                continue                      # no default to fall back to; null is a real error
            annotation = str(getattr(field, "annotation", ""))
            if "Optional" not in annotation and "None" not in annotation:
                offenders.append(f"{tool.name}.{fname}: {annotation}")
    assert not offenders, (
        "these optional parameters still reject an explicit null, so a model that fills every "
        f"slot cannot call them: {offenders[:12]}")


def test_embed_region_specifically():
    """Named because it is the one the sweep caught."""
    from agent_runtime.rs_embed_tools import make_rs_embed_tools
    tool = {t.name: t for t in make_rs_embed_tools()}["embed_region"]
    out = tool.invoke({"file_id": "missing", "models": "gse", "buffer_m": None, "lon": None,
                       "lat": None, "bbox": None, "start": None, "end": None, "name": "box"})
    # It should fail on the BAD FILE ID, which is a real problem, not on the nulls.
    assert "validation" not in str(out).lower()
    assert "extent" in str(out).lower() or "missing" in str(out).lower()
