#!/usr/bin/env bash
# Run the agent API locally for development and for driving the I-GUIDE prototype.
#
# Every feature must be exercised through the prototype UI, which needs a reachable
# agent API. This script is the one supported way to start it locally, so the env
# contract lives in exactly one place.
#
#   AGENT_CHAT_API_KEY        set it -> the prototype must send a matching X-API-KEY.
#                             unset  -> auth fails CLOSED (500) unless the next var is on.
#   AGENT_CHAT_AUTH_OPTIONAL  =1 to run with no auth at all (local only).
#   AGENT_CORS_ORIGINS        must include the prototype's origin, e.g.
#                             http://localhost:8131, or the browser blocks the request.
#   AGENT_KB_BACKEND          defaults to `local` in agent_kb.py so tests stay offline. This
#                             script sets it to `opensearch` when OPENSEARCH_NODE is present,
#                             because a dev server reading the file-backed store instead of the
#                             indexed corpus looks like a retrieval-quality problem, not a
#                             config one. Set it yourself to override.
#
# Usage:
#   scripts/run_agent_api_dev.sh                 # keyed: dev-key, CORS for :8131
#   AGENT_CHAT_AUTH_OPTIONAL=1 scripts/run_agent_api_dev.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Backends (OpenSearch, LLM keys) live in .env. This worktree may not have one; fall
# back to the primary checkout rather than starting with a half-configured server.
#
# IMPORTANT: an explicitly exported variable must WIN over the file. `set -a; . .env`
# does the opposite — it clobbered PORT=5002 with the file's 3500 and replaced the
# API key the caller asked for, silently. That is the shell twin of the
# load_dotenv(override=True) problem elsewhere in this repo: the file beating the
# environment makes a per-run override impossible and the reason invisible.
load_env_without_override() {
  local file="$1" key val
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"; key="${key#export }"; key="$(printf '%s' "$key" | tr -d '[:space:]')"
    case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
    # already set and non-empty in the environment -> the caller wins, skip the file
    if [ -n "${!key:-}" ]; then continue; fi
    val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    export "$key=$val"
  done < "$file"
}

for candidate in "$REPO/.env" "/Users/yfkang/i-guide-platform-flask-servers/.env"; do
  if [ -f "$candidate" ]; then
    echo "[run_agent_api_dev] loading env from $candidate (existing vars win)"
    load_env_without_override "$candidate"
    break
  fi
done

export PORT="${PORT:-5002}"

# The agent KB defaults to the LOCAL file-backed store (agent_kb.py) so tests and offline runs
# never reach the cluster — that default is right for them and wrong for a dev server driving the
# prototype. Measured: a live turn logged `agent_kb_search -> no results` while the same query
# against the cluster returned 8, and nothing anywhere reported which store had been read.
if [ -n "${OPENSEARCH_NODE:-}" ]; then
  export AGENT_KB_BACKEND="${AGENT_KB_BACKEND:-opensearch}"
fi
echo "[run_agent_api_dev] agent KB backend: ${AGENT_KB_BACKEND:-local (no OPENSEARCH_NODE)}"

# The method library is mounted read-only into the sandbox and read agent-side for
# kb_method_search. Without this, both silently report an empty library.
if [ -z "${AGENT_METHOD_LIBRARY_DIR:-}" ] && [ -d "$REPO/agent_chat_files/method_library" ]; then
  export AGENT_METHOD_LIBRARY_DIR="$REPO/agent_chat_files/method_library"
fi
echo "[run_agent_api_dev] method library: ${AGENT_METHOD_LIBRARY_DIR:-none built}"
export AGENT_CHAT_API_KEY="${AGENT_CHAT_API_KEY:-dev-key}"
export AGENT_CORS_ORIGINS="${AGENT_CORS_ORIGINS:-http://localhost:8131,http://127.0.0.1:8131}"
# The prototype runs outside the compose network, so the in-container embedding
# hostname is unreachable here; prefer an explicitly provided value.
export FLASK_EMBEDDING_URL="${FLASK_EMBEDDING_URL_LOCAL:-${FLASK_EMBEDDING_URL:-}}"

if [ "${AGENT_CHAT_AUTH_OPTIONAL:-}" = "1" ]; then
  echo "[run_agent_api_dev] AUTH DISABLED (AGENT_CHAT_AUTH_OPTIONAL=1)"
else
  echo "[run_agent_api_dev] auth ON — send X-API-KEY: $AGENT_CHAT_API_KEY"
fi
echo "[run_agent_api_dev] CORS origins: $AGENT_CORS_ORIGINS"
echo "[run_agent_api_dev] listening on http://127.0.0.1:$PORT"

exec python3 -m api.server
