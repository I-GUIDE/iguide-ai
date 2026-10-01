---
name: reasoning-models-lose-their-thinking
description: gpt-oss:120b repeated tool calls because langchain_openai drops reasoning_content; fixed by promoting it into content on tool-calling steps
metadata: 
  node_type: memory
  type: project
  originSessionId: 9b1d4b74-150f-4936-b715-5b9f69c0880d
  modified: 2026-09-01T15:47:47.613Z
---

Measured 2026-09-01. `gpt-oss:120b` on AnvilGPT answers a tool-calling step with
`content: None`, `reasoning_content: "<the plan>"` and the tool call. **langchain_openai 1.6
parses neither reasoning field onto the message** — not `additional_kwargs` (which holds only
`['refusal']`), not `response_metadata` — so the assistant turn replayed on the next step is
`{role: assistant, content: None, tool_calls: [...]}`.

**Why:** the model has no record of why it called the tool, re-derives from the user request,
and issues the same call again. Observed: five `admin_boundary` calls in one turn across two
peers (first two with IDENTICAL args), then `execute_code` cycling through guesses at a
filename it already had.

**This is a SHIM defect, not a reasoning-model defect** (corrected 2026-09-01 after the user
pointed out gpt-5.6-luna is OpenAI — my first probe wrongly pointed luna at the AnvilGPT base
URL and 404'd, and I believed my own broken test). Measured against api.openai.com with tools
bound: OpenAI NEVER returns reasoning text over chat/completions. Message keys are identical
for gpt-4o, luna, gpt-5.2 and o4-mini, and `reasoning_content` is absent — not null — even
when the model demonstrably reasoned (gpt-5.2 at effort=high burns 128–214 reasoning tokens
and returns none of the text). gpt-4o is NOT immune "because its plan lives in content": it
also returns `content=None` on a tool-calling step. And luna is a special case — with function
tools OpenAI hard-400s any `reasoning_effort` but `'none'`, `resolve_effort` forces it, and at
`'none'` luna emits `reasoning_tokens=0`. **As this agent calls it, luna is a non-reasoning
tool-caller.** Only `/v1/responses` carries reasoning across steps; we do not use it, and
switching would change endpoint/content-blocks/streaming for every OpenAI turn.

**How to apply:** `_reasoning_preserving_chat_openai()` in `agent_runtime/executor_factory.py`
subclasses ChatOpenAI and, in `_create_chat_result`, stashes the reasoning and promotes it into
`content` **only when the message also has tool_calls** — promoting it on a final message would
leak deliberation into the user's answer. Both construction paths use it, including the
per-request one (the model picker). After the fix plus the `admin_boundary` suffix fix, the
same query used ONE `admin_boundary` call instead of three. If another local/OSS reasoning
model behaves oddly, check the raw payload for `reasoning_content` before blaming the prompt.
See [[keep-capability-atlas-current]], [[verify-in-web-prototype]].
