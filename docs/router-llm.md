# LLM Router - Intelligent Search Module Selection

An LLM-powered routing system that intelligently decides which search modules to use based on the query. Uses the same AnvilGPT API as your generation pipeline.

## Quick Start

```python
from rag_pipeline.router_llm import build_default_registry, LLMRouter

# Wire your search backends
registry = build_default_registry(
    keyword_impl=your_keyword_search,
    semantic_impl=your_semantic_search,
    spatial_impl=your_spatial_search,
    graph_impl=your_neo4j_search,
)

# Create router (uses AnvilGPT automatically)
router = LLMRouter(registry)

# Plan and execute
plan = router.plan("Find weather stations near Chicago")
results = router.execute(plan, ctx={"state": state_dict})
```

## How It Works

1. **Always runs keyword search**
2. **LLM decides** whether to add semantic, spatial, or graph search
3. **Executes selected modules** with real backends
4. **Gracefully falls back** to keyword-only if LLM fails

## Configuration

Uses your existing AnvilGPT setup:

```bash
# Required (already configured for generation.py)
ANVILGPT_URL=https://anvilgpt.rcac.purdue.edu/api/chat/completions
ANVILGPT_KEY=your_api_key
ANVILGPT_MODEL=gpt-oss:120b  # optional

# Optional backend flags
SEMANTIC_BACKEND_ENABLED=1  # default: 1
SPATIAL_BACKEND_ENABLED=1   # default: 0
NEO4J_URI=bolt://localhost:7687
```

## File Structure

```
rag_pipeline/
├── router_llm.py                    # Core router implementation
├── tests/
│   ├── test_router_llm.py           # Unit tests (4 tests)
│   └── test_e2e_llm_router.py       # End-to-end pipeline test
├── routing.py                       # Original router (unchanged)
├── generation.py                    # Uses AnvilGPT (unchanged)
└── llm_utils.py                     # Shared AnvilGPT client (unchanged)
```

## Running Tests

```bash
# Unit tests
pytest rag_pipeline/tests/test_router_llm.py -v

# Full pipeline test with real data
python rag_pipeline/tests/test_e2e_llm_router.py
```

## Integration Example

Add a feature flag to switch between routers:

```python
import os
from rag_pipeline.routing import rag_pipeline  # original

# In your routing logic:
use_llm_router = os.getenv("USE_LLM_ROUTER", "0") == "1"

if use_llm_router:
    from rag_pipeline.router_llm import build_default_registry, LLMRouter
    
    registry = build_default_registry(
        keyword_impl=lambda q, c: retrieve_keyword(c.get("state", {})),
        semantic_impl=lambda q, c: retrieve_semantic(c.get("state", {})),
        spatial_impl=lambda q, c: retrieve_spatial(c.get("state", {})),
        graph_impl=lambda q, c: retrieve_neo4j(c.get("state", {})),
    )
    
    router = LLMRouter(registry)
    plan = router.plan(query)
    router_output = router.execute(plan, ctx={"state": state})
    
    # Merge results into state
    for result in router_output["results"]:
        merge_retrieval(state, source=result["module"], 
                       hits=result["items"], limit=top_k)
else:
    # Use original pipeline
    state = rag_pipeline(state)
```

## Example Routing Decisions

| Query | Modules | Rationale |
|-------|---------|-----------|
| "climate datasets" | keyword | Simple keyword match sufficient |
| "studies about carbon emissions" | keyword, semantic | 'about' suggests conceptual search |
| "sensors within 10 miles of Chicago" | keyword, semantic, spatial | Location + distance detected |
| "organizations connected to NCSA" | keyword, semantic, graph | Entity relationships detected |

## Architecture

- **`AnvilGPTClient`** - Uses `rag_pipeline.llm_utils.call_llm()` (same as generation)
- **`LLMDecisionEngine`** - Caches routing decisions (5 min TTL)
- **`ModuleRegistry`** - Plugin system for search backends
- **`LLMRouter`** - Orchestrates planning and execution

## Adding New Search Modules

```python
class MySearchAdapter:
    name = "my_search"
    
    def is_available(self) -> bool:
        return os.getenv("MY_BACKEND_ENABLED") == "1"
    
    def supports(self, query: str) -> bool:
        return True  # or add heuristics
    
    def execute(self, query: str, ctx: dict) -> dict:
        hits = my_search_function(query, ctx)
        return {
            "module": self.name,
            "items": hits,
            "meta": {"engine": "my_engine"}
        }

# Register it
registry.register(MySearchAdapter())
```

## Benefits

✅ **Query-aware routing** - LLM decides based on query semantics  
✅ **Cost-efficient** - Only runs expensive searches when needed  
✅ **Shared infrastructure** - Reuses AnvilGPT setup  
✅ **Graceful fallbacks** - Never breaks the pipeline  
✅ **Transparent** - Full rationale for each decision  
✅ **Testable** - Mock LLM for deterministic tests  

## Troubleshooting

**Router always returns keyword-only:**
- This is expected if ANVILGPT_KEY not set (safe fallback)
- Check ANVILGPT_URL and ANVILGPT_KEY are correct

**Module not included:**
- Check `is_available()` returns True
- Verify backend env vars (SPATIAL_BACKEND_ENABLED, NEO4J_URI, etc.)

**Tests fail:**
- Run: `pytest rag_pipeline/tests/test_router_llm.py -v`
- Check imports after file reorganization
