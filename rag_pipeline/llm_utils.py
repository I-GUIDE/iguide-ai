from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

_llm_callable: Optional[Callable[[str], str]] = None


def _completion_url() -> str:
    base = (os.getenv("VLLM_PROXY") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    lowered = base.lower()
    if lowered.endswith("/chat/completions"):
        return base
    if lowered.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def register_llm_callable(func: Callable[[str], str]) -> None:
    """
    Allow applications to register a synchronous LLM callable.
    This is kept for backward compatibility with tests.
    """
    global _llm_callable
    _llm_callable = func


def call_llm(prompt: str) -> str:
    """
    Send a prompt to an OpenAI-compatible chat completions endpoint and return text.
    
    Uses environment variables:
    - VLLM_API_KEY: Preferred API key
    - VLLM_PROXY: Preferred base/completions URL
    - VLLM_MODEL: Preferred model
    - OPENAI_KEY / OPENAI_BASE_URL / OPENAI_CHAT_MODEL / OPENAI_MODEL as fallback
    
    Returns the generated text response from the model.
    Raises RuntimeError on configuration or API errors.
    """
    # Allow test override via register_llm_callable
    if _llm_callable is not None:
        result = _llm_callable(prompt)
        if isinstance(result, str):
            return result
        logger.warning("LLM callable returned non-string result; coercing to string.")
        return str(result)
        # An exception from the injected callable PROPAGATES, exactly as one from the real
        # provider does. It used to be caught here and returned as
        # "I could not compose an answer due to a generation error." — so the test seam behaved
        # differently from production in the one respect a failure test cares about.
        #
        # The consequence: `publication_extractor` distinguishes `llm_unavailable` (the call
        # raised) from `llm_unparseable` (it answered with something unusable), and through this
        # seam the first was UNREACHABLE — a raising double always arrived as an unparseable
        # string. The degradation path the plan wanted exercisable offline could not be exercised
        # at all, and a test that "covered" it was testing the canned string.
        #
        # The canned strings further down are a different case and stay: an empty or malformed
        # response from a reachable provider should degrade to text rather than crash the
        # answer path.
    
    # LLM_PROVIDER=claude-cli routes through the local `claude` executable so extraction
    # batches and eval sweeps cost nothing during development. Dev-only by contract; the
    # backend itself refuses to run where a deployment marker is present.
    from . import llm_claude_cli

    if llm_claude_cli.is_selected():
        llm_claude_cli.check_not_deployed()
        return llm_claude_cli.call(prompt)

    # Production path: OpenAI-compatible call
    url = _completion_url()
    key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_KEY")
    
    if not url or not key:
        raise RuntimeError(
            "❌ Missing VLLM_API_KEY (or OPENAI_KEY) environment variable. "
            "Please set it in your .env file."
        )
    
    model = (
        os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "Qwen/Qwen3.5-9B"
    )
    
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    
    try:
        logger.debug(f"Calling LLM at {url} with model {model}")
        response = requests.post(url, headers=headers, json=body, timeout=60)
        
        if response.status_code == 403:
            raise RuntimeError("🚫 403 Forbidden: Invalid or expired API key or endpoint.")
        
        if response.status_code == 401:
            raise RuntimeError("🚫 401 Unauthorized: Invalid API key.")
        
        if response.status_code != 200:
            error_text = response.text[:200]
            raise RuntimeError(f"⚠️ HTTP {response.status_code}: {error_text}")
        
        data = response.json()
        if data is None:
            logger.error("LLM endpoint returned HTTP 200 with null JSON body.")
            return "I could not compose an answer due to an empty LLM response."
        
        # OpenAI-compatible shape
        if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            
            # Handle reasoning mode: if content is None but reasoning_content exists, use that
            if content is None:
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    logger.warning(
                        f"AnvilGPT returned None content but reasoning_content exists "
                        f"(finish_reason: {choice.get('finish_reason')}). Using reasoning_content."
                    )
                    content = reasoning_content
                else:
                    # Check provider_specific_fields as fallback
                    provider_fields = message.get("provider_specific_fields", {})
                    reasoning_content = provider_fields.get("reasoning_content")
                    if reasoning_content:
                        logger.warning(
                            f"AnvilGPT returned None content but reasoning_content in provider_specific_fields "
                            f"(finish_reason: {choice.get('finish_reason')}). Using reasoning_content."
                        )
                        content = reasoning_content
            
            if content is None:
                finish_reason = choice.get("finish_reason", "unknown")
                raise RuntimeError(
                    f"⚠️ AnvilGPT returned empty content (finish_reason: {finish_reason}). "
                    f"Response may have been truncated. Full response: {str(data)[:500]}"
                )
            
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                logger.warning(
                    f"⚠️ AnvilGPT response was truncated (hit token limit). "
                    f"Received {len(content)} characters."
                )
            
            logger.debug(f"LLM response received: {len(content)} characters")
            return content

        # Ollama-style shape fallback: {"message": {"content": "..."}}
        if isinstance(data, dict):
            message = data.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    logger.debug("AnvilGPT response received via ollama-style payload.")
                    return content

        logger.error("Unexpected response format from LLM endpoint: %s", str(data)[:500])
        return "I could not compose an answer due to an unexpected LLM response format."
    
    except requests.RequestException as exc:
        logger.error(f"LLM request failed: {exc}")
        raise RuntimeError(f"🔌 LLM request failed: {exc}")


__all__ = ["call_llm", "register_llm_callable"]
