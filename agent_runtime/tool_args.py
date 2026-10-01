"""Let an explicit ``null`` stand for "use the default" on every optional tool parameter.

A model writing tool arguments tends to fill EVERY slot the schema offers, putting ``null`` in
the ones it does not need. Where a parameter is annotated ``str`` with a default, pydantic
refuses that call before the tool ever runs, and the model learns nothing from the schema error
except to try again. Measured live on gpt-oss:120b::

    embed_region({'file_id': ..., 'model': 'gse', 'buffer_m': None, 'lon': None, 'lat': None,
                  'bbox': None, 'start': None, 'end': None, ...})
    ValidationError: 3 validation errors for embed_region

Nothing was wrong with that call. ``start=None`` means "no opinion about the date window", which
is exactly what the default expresses — the tool had a perfectly good answer and refused to use
it over a type annotation.

A scan found **141 such parameters across 62 of the 80 exposed tools**, so this is not a handful
of signatures to patch. It also cannot be fixed by wrapping the function body: pydantic validates
against the SCHEMA, which LangChain infers from the signature, so the call is rejected before any
of our code runs. The signature itself has to say the parameter is nullable.

So the wrapper rewrites the signature — every defaulted parameter becomes ``Optional[...]`` — and
substitutes the original default when a null arrives. A real value still wins, and a parameter
that was always required stays required: this widens what is ACCEPTED and changes nothing about
what the tool then does.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional

_EMPTY = inspect.Parameter.empty


def accept_null_defaults(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return *func* with every defaulted parameter made nullable, null meaning "the default".

    Untouched: parameters with no default (still required), parameters whose default is already
    None (already nullable), and anything unannotated (nothing to widen).
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins and C functions
        return func

    hints = dict(getattr(func, "__annotations__", {}))
    defaults: dict = {}
    params = []
    for name, param in signature.parameters.items():
        if (param.default is _EMPTY or param.default is None
                or param.annotation is _EMPTY
                or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD)):
            params.append(param)
            continue
        defaults[name] = param.default
        params.append(param.replace(annotation=Optional[param.annotation]))
        if name in hints:
            hints[name] = Optional[hints[name]]

    if not defaults:
        return func

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        for key, value in list(kwargs.items()):
            if value is None and key in defaults:
                kwargs[key] = defaults[key]
        return func(*args, **kwargs)

    # LangChain infers the args schema from these two, which is why setting them is the fix
    # rather than a cosmetic touch-up.
    wrapper.__signature__ = signature.replace(parameters=params)  # type: ignore[attr-defined]
    wrapper.__annotations__ = hints
    return wrapper
