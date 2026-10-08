"""Static analysis for extracted knowledge elements. Pure: no I/O, no network, no execution.

Everything here works on source text and ``ast``/``symtable`` only, so it is fully testable
without a cluster, an LLM, or a sandbox — which is deliberate: this is the layer the
correctness of every extracted unit rests on.
"""

from .callability import analyze_module, analyze_unit, iter_units, module_scope
from .signatures import contract_params, params_of, signature_of
from .slices import (build_module_source, build_unit_slice, has_module_side_effects,
                     slice_sha)

__all__ = ["analyze_module", "analyze_unit", "iter_units", "module_scope",
           "signature_of", "params_of", "contract_params",
           "build_module_source", "build_unit_slice", "slice_sha",
           "has_module_side_effects"]
