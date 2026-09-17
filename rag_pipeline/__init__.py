from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from importlib import import_module

__all__ = [
    "make_langchain_granular_tools",
    "make_langchain_mcp_tools",
    "run_agent_chat",
    "build_agent_executor",
    "build_code_agent_executor",
    "run_agent_query",
]

_EXPORT_MAP = {
    "make_langchain_granular_tools": (".langchain_granular_tools", "make_langchain_granular_tools"),
    "make_langchain_mcp_tools": (".langchain_mcp_tools", "make_langchain_mcp_tools"),
    "run_agent_chat": (".agent_chat_service", "run_agent_chat"),
    "build_agent_executor": ("agent_runtime.executor_factory", "build_agent_executor"),
    "build_code_agent_executor": ("agent_runtime.executor_factory", "build_code_agent_executor"),
    "run_agent_query": ("agent_runtime.graph_runtime", "run_agent_query"),
}


def __getattr__(name: str):
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'rag_pipeline' has no attribute '{name}'")
    module_name, attr_name = target
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
