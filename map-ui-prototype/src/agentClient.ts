// Real client for the I-GUIDE agent SSE API (see api/server.py :: /agent/chat/stream
// and examples/iguide_chat_prototype.html). Event-driven: per-event hooks fire live
// so the map/chat update AS the agent works; onFinal reconciles against the
// authoritative terminal `result`. This module IS the swap point that replaces the
// local deterministic agentBrain.
import { authErrorFrom, isAuthStatus, setRefreshUrl, setSigninUrl, withTokenRetry } from './auth';
export interface AgentConfig {
  endpoint: string;        // .../agent/chat/stream
  uploadEndpoint: string;  // .../agent/files/upload
  apiKey: string;
  /** Selected model, e.g. 'gpt-4o-2024-11-20' or 'qwen3.6:27b'. Empty = the agent's default. */
  model?: string;
  /** 'openai' | 'anvilgpt'. Empty lets the server infer it from the model id. */
  provider?: string;
  /** Reasoning models only (gpt-5.x, o-series): 'none'|'low'|'medium'|'high'|'xhigh'. */
  reasoningEffort?: string;
  /** Which agent writes the CODE — a different axis from `model`, which is what
   *  writes the ANSWER. 'langchain' | 'claude' | 'opencode'. Empty = server default. */
  codePeer?: string;
  /** Model for a CLI code peer, e.g. 'sonnet' | 'opus'. Empty = that peer's default.
   *  Independent of `model`, which is what writes the ANSWER. */
  codePeerModel?: string;
  /** Orchestration shape: '' = deployment default, 'peers' = supervisor routes between a
   *  search peer and an analyze peer, 'unified' = one agent does both in one context.
   *  Per-request so both shapes can be compared side by side in separate tabs. */
  orchestration?: string;
}

export interface ModelCatalogue {
  default: { provider: string; model: string };
  /** Accepted reasoning_effort values, per the API's own error message. */
  reasoning_efforts?: string[];
  providers: { provider: string; label: string; configured: boolean;
               models: string[]; stale?: boolean;
               /** Env var this provider is waiting on, when not configured. */
               needs?: string;
               /** Configured, but with strings attached — shown beside the group label. */
               caveat?: string;
               /** Shown with the group, e.g. to separate answering models from peer models. */
               note?: string;
               /** Subset of `models` that accept reasoning_effort. */
               reasoning_models?: string[];
               /** Legal reasoning_effort values PER model, with function tools attached.
                *  A global list offered 'high' on models that refuse any real level. */
               effort_options?: Record<string, string[]>;
               /** Models that REFUSE tools unless this exact value is sent (gpt-5.6-*). */
               effort_required?: Record<string, string> }[];
  /** The code-peer backends a request may select. A second axis, reported under its
   *  own key so nothing conflates "which model answers" with "which agent codes". */
  code_peers?: {
    default: string;
    peers: { id: string; label: string; available: boolean;
             /** Why not, when unavailable: 'no credential' | 'image not built'. */
             reason?: string | null;
             model?: string; auth?: string | null;
             /** Selectable models for this peer, as aliases. Absent = not selectable. */
             models?: string[] }[];
  };
}

export interface UiConfig {
  /** What this deployment is for. Absent on a server built before modes existed. */
  mode?: 'dev' | 'demo' | 'token';
  demo_mode: boolean;
  api_key_required: boolean;
  /** Token mode only: where the BROWSER refreshes an aged-out access cookie, and where it
   *  sends someone who is not signed in. Reported by the server rather than compiled in, so
   *  one bundle runs against either tier — dev and production are different backends. */
  refresh_url?: string;
  signin_url?: string;
}

/** Ask the deployment whether it is open, BEFORE trying to authenticate against it.
 *
 * Unauthenticated on purpose at both ends: a client that does not have a key is exactly the
 * client that needs to know whether it needs one. Failure degrades to "not a demo", which keeps
 * the settings reachable — the safe direction to be wrong in, since the alternative is a page
 * with no way to enter a credential it turns out to need. */
export async function fetchUiConfig(cfg: AgentConfig): Promise<UiConfig | null> {
  try {
    const r = await fetch(absoluteUrl('/agent/ui-config', cfg));
    if (!r.ok) return null;
    const parsed = (await r.json()) as UiConfig;
    // Learned here and nowhere else: every later 401 depends on knowing where to refresh, and
    // this is the one call that happens before the page can authenticate at all.
    setRefreshUrl(parsed.refresh_url);
    setSigninUrl(parsed.signin_url);
    return parsed;
  } catch {
    return null;
  }
}

export interface WhoAmI {
  mode: string;
  signedIn: boolean;
  user: { id: string; role: number } | null;
  permitted: boolean;
  reason: string | null;
  requiredRole?: number;
}

/** Who the SERVER thinks we are. The cookie is httpOnly, so the browser cannot answer this
 *  itself — it has to ask. Used to scope stored conversations to their owner and, later, to
 *  render a profile. Degrades to "nobody", which lists nothing rather than everything. */
export async function fetchWhoAmI(cfg: AgentConfig): Promise<WhoAmI | null> {
  try {
    const r = await fetch(absoluteUrl('/agent/whoami', cfg), { credentials: CREDENTIALS });
    if (!r.ok) return null;
    return (await r.json()) as WhoAmI;
  } catch {
    return null;
  }
}

export interface ConversationSummary {
  memoryId: string;
  conversationName?: string;
  createdAt?: string;
  updatedAt?: string;
  threadId?: string;
  messageCount?: number;
  layerCount?: number;
  fileCount?: number;
}

/** THIS USER's conversations, from the server. Summaries only — never transcripts. */
export async function listConversations(cfg: AgentConfig): Promise<ConversationSummary[] | null> {
  try {
    return await withTokenRetry(async () => {
      const r = await fetch(absoluteUrl('/agent/conversations', cfg), { credentials: CREDENTIALS });
      if (isAuthStatus(r.status)) throw await authErrorFrom(r);
      if (!r.ok) return null;
      const body = await r.json();
      return (body?.conversations || []) as ConversationSummary[];
    });
  } catch {
    // null, NOT []: "could not ask" and "you have none" look identical to a caller otherwise,
    // and the first must not silently present as an empty history.
    return null;
  }
}

/** Store this conversation server-side, so it follows the user rather than the browser. */
export async function putConversation(cfg: AgentConfig, memoryId: string,
                                      record: unknown): Promise<boolean> {
  try {
    return await withTokenRetry(async () => {
      const r = await fetch(absoluteUrl(`/agent/conversations/${encodeURIComponent(memoryId)}`, cfg), {
        method: 'PUT', headers: authHeaders(cfg, true), credentials: CREDENTIALS,
        body: JSON.stringify(record),
      });
      if (isAuthStatus(r.status)) throw await authErrorFrom(r);
      return r.ok;
    });
  } catch {
    return false;
  }
}

/** Read one conversation back. */
export async function getConversation(cfg: AgentConfig, memoryId: string): Promise<any | null> {
  try {
    return await withTokenRetry(async () => {
      const r = await fetch(absoluteUrl(`/agent/conversations/${encodeURIComponent(memoryId)}`, cfg),
                            { credentials: CREDENTIALS });
      if (isAuthStatus(r.status)) throw await authErrorFrom(r);
      if (!r.ok) return null;
      return await r.json();
    });
  } catch {
    return null;
  }
}

/** Ask the agent which models a request may select. */
export async function fetchModels(cfg: AgentConfig): Promise<ModelCatalogue | null> {
  try {
    // apiBase is only the ORIGIN, so name the path explicitly — the same way download
    // URLs are resolved. new URL('models', origin) would hit /models.
    const url = absoluteUrl('/agent/models', cfg);
    const r = await fetch(url, { headers: authHeaders(cfg, false), credentials: CREDENTIALS });
    if (!r.ok) return null;
    return (await r.json()) as ModelCatalogue;
  } catch {
    return null;   // the picker degrades to "agent default" rather than blocking the page
  }
}

export interface FileRecord {
  file_id: string;
  filename: string;
  download_url: string;
  kind: string;
}

export interface TraceLine { text: string; kind?: string }

export interface StreamHandlers {
  onTrace?: (line: TraceLine) => void;                 // status / routing / reasoning
  onToolCall?: (name: string, args: any) => void;      // e.g. spatial_search({bbox})
  onToolResult?: (name: string, parsed: any, raw: any) => void;
  onFile?: (files: FileRecord[]) => void;              // artifacts as they appear (deduped)
  onMapLayer?: (layer: MapLayerEvent) => void;         // untruncated geometry to plot live
  onAnswerChunk?: (text: string) => void;              // answer text when it arrives
  onIds?: (ids: { threadId?: string; memoryId?: string }) => void;
}

export interface MapLayerEvent {
  id: string;
  source: string;
  label: string;
  count?: number;
  geojson?: import('geojson').FeatureCollection;
  url?: string;                 // fetch instead of inlining (large layers)
  render?: string;              // 'heatmap' | 'choropleth' | 'categories' | 'points' | 'shapes'
  outline?: boolean;            // boundary only, so anything drawn beneath stays visible
  styleBy?: string;             // property to shade by (numeric, or a class name for 'categories')
  legend?: { label: string; color: [number, number, number, number] }[];
  style_by?: string;            // snake_case as the agent sends it
  sampled?: boolean;            // true when the layer is a subset of the data
  total?: number;               // full population size when sampled
  bounds?: [number, number, number, number];  // raster footprint [minLon,minLat,maxLon,maxLat]
  opacity?: number;             // raster draping opacity
  // The vectors this picture was made FROM. An embedding raster is a 3-colour projection and
  // answers nothing by itself; every later question about it ("predict from this layer",
  // "compare it with that one") needs the real package. Carried on the layer so it survives
  // into the session store and outlives the turn that produced it.
  embedding?: EmbeddingRef;
}

export interface EmbeddingRef {
  file_id: string;
  filename?: string;
  model?: string;               // which model inside the package this layer draws
  months?: string;
  models_in_package?: string[];
  recoloured_on_shared_basis?: boolean;
}

export interface StreamResult {
  answer: string;
  response: any;            // terminal result payload (source of truth)
  downloads: FileRecord[];
  threadId?: string;
  memoryId?: string;
  error?: string;
}

export function apiBase(cfg: AgentConfig): string {
  for (const v of [cfg.endpoint, cfg.uploadEndpoint]) {
    if (/^https?:\/\//i.test(v)) { try { return new URL(v).origin; } catch { /* */ } }
  }
  return location.origin;
}

export function absoluteUrl(path: string, cfg: AgentConfig): string {
  if (!path) return '#';
  const p = String(path).trim().replace(/^sandbox:/i, '');
  if (/^https?:\/\//i.test(p)) return p;
  try { return new URL(p, apiBase(cfg)).toString(); } catch { return '#'; }
}

/** Send cookies on every agent call.
 *
 *  Same-origin would send them anyway — the UI and the API share agent.i-guide.io — but a
 *  developer running `npm run dev` against the deployed API is cross-origin, and there the
 *  default omits the cookie and every request looks unauthenticated for no visible reason. */
const CREDENTIALS: RequestCredentials = 'include';

function authHeaders(cfg: AgentConfig, json: boolean): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h['Content-Type'] = 'application/json';
  if (cfg.apiKey.trim()) h['X-API-KEY'] = cfg.apiKey.trim();
  return h;
}

export function newThreadId(): string {
  try { if (crypto?.randomUUID) return 'sess-' + crypto.randomUUID(); } catch { /* */ }
  return 'sess-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}

// Recursively harvest {file_id, filename, download_url} records from any payload shape.
function collectDownloads(value: any, into: Map<string, FileRecord>): void {
  if (!value) return;
  if (Array.isArray(value)) { value.forEach((v) => collectDownloads(v, into)); return; }
  if (typeof value === 'string') { try { collectDownloads(JSON.parse(value), into); } catch { /* */ } return; }
  if (typeof value !== 'object') return;
  if (value.download_url && (value.file_id || value.filename)) {
    into.set(String(value.file_id || value.download_url), {
      file_id: value.file_id || '', filename: value.filename || 'download',
      download_url: value.download_url, kind: value.kind || '',
    });
  }
  Object.values(value).forEach((v) => collectDownloads(v, into));
}

function parseMaybeJson(raw: any): any {
  if (raw == null) return null;
  if (typeof raw !== 'string') return raw;
  // Tool content is often a LangChain ToolMessage repr: content='{...}' name=... ; extract the JSON.
  const m = raw.match(/content='([\s\S]*?)'\s+name=/) || raw.match(/^\s*(\{[\s\S]*\})\s*$/);
  const candidate = m ? m[1] : raw;
  try { return JSON.parse(candidate); } catch { return null; }
}

export async function uploadFiles(files: File[], cfg: AgentConfig): Promise<FileRecord[]> {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f, f.name));
  const res = await withTokenRetry(async () => {
    const r = await fetch(cfg.uploadEndpoint, {
      method: 'POST', headers: authHeaders(cfg, false), credentials: CREDENTIALS, body: fd });
    if (isAuthStatus(r.status)) throw await authErrorFrom(r);
    if (!r.ok) throw new Error(await describeError(r));
    return r;
  });
  const json = await res.json();
  return (json.files || []) as FileRecord[];
}

async function describeError(res: Response): Promise<string> {
  const raw = await res.text().catch(() => '');
  try { const j = JSON.parse(raw); return `HTTP ${res.status}: ${j.error || j.message || raw}`; }
  catch { return `HTTP ${res.status}: ${raw || res.statusText}`; }
}

export interface StreamOpts {
  fileIds?: string[];
  threadId: string;
  memoryId?: string | null;
  agentDev?: boolean;
  signal?: AbortSignal;
  conversationName?: string;
  includeMcpTools?: boolean;              // spatial toggle -> MCP geo tools on/off
  enabledSearchMethods?: string[] | null; // spatial toggle -> restrict retrieval tools
}

export async function streamChat(
  text: string,
  opts: StreamOpts,
  cfg: AgentConfig,
  h: StreamHandlers,
): Promise<StreamResult> {
  const payload = {
    user_input: text,
    memory_id: opts.memoryId ?? null,
    thread_id: opts.threadId,
    conversation_name: opts.conversationName || 'I-GUIDE map UI',
    tool_strategy: 'granular',
    use_persistent_memory: true,
    smart_tool_routing: true,
    include_mcp_tools: opts.includeMcpTools ?? true,
    code_exec: true,
    agent_dev: opts.agentDev ?? true, // detailed trace so the map can react to tool events
    file_ids: (opts.fileIds || []).filter((id) => id && !id.startsWith('tmp:')),
    enabled_search_methods: opts.enabledSearchMethods ?? null,
    verbose: false,
    // Only sent when chosen: absent model AND provider means the agent uses its configured
    // default (OpenAI gpt-4o here), which is what every older client does.
    ...(cfg.model ? { model: cfg.model } : {}),
    ...(cfg.provider ? { provider: cfg.provider } : {}),
    ...(cfg.reasoningEffort ? { reasoning_effort: cfg.reasoningEffort } : {}),
    // Absent = the deployment's AGENT_CODE_PEER default, so a client that never
    // sets it behaves exactly as it did before the control existed.
    ...(cfg.codePeer ? { code_peer: cfg.codePeer } : {}),
    ...(cfg.codePeerModel ? { code_peer_model: cfg.codePeerModel } : {}),
    // Absent = the deployment's AGENT_UNIFIED_PEER default. Sent as a boolean because the
    // server reads it as one; the two named options exist so the UI can say what they mean.
    ...(cfg.orchestration ? { unifiedPeer: cfg.orchestration === 'unified' } : {}),
  };

  // A thunk, not a Promise: a retry has to ISSUE a new POST, and a started request cannot be
  // re-issued. Only an expired token retries — see withTokenRetry.
  const body = await withTokenRetry(async () => {
    const r = await fetch(cfg.endpoint, {
      method: 'POST', headers: authHeaders(cfg, true), credentials: CREDENTIALS,
      body: JSON.stringify(payload), signal: opts.signal,
    });
    if (isAuthStatus(r.status)) throw await authErrorFrom(r);
    if (!r.ok || !r.body) throw new Error(await describeError(r));
    return r.body;      // narrowed here; a Response would lose it crossing the await
  });

  const downloads = new Map<string, FileRecord>();
  // Tool calls whose result has not arrived. Only used to decide whether a result row has to
  // name its tool: with one call outstanding the indent under the call above is unambiguous,
  // with four it is a guess. A name is deleted on its result, so a tool called twice in one
  // batch collapses to one entry — which under-reports rather than mislabels.
  const pending = new Set<string>();
  // How many calls the widest point of the current batch held. Naming only while calls are
  // still outstanding left the LAST result of a batch bare — three rows named and one not,
  // which reads as an oversight and makes the reader infer the odd one out. A batch is named
  // in full or not at all, and the counter resets when the batch drains.
  let batchWidth = 0;
  const state: StreamResult = { answer: '', response: null, downloads: [], threadId: opts.threadId, memoryId: opts.memoryId ?? undefined };
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  const handleBlock = (block: string) => {
    if (!block.trim()) return;
    let eventName = 'message';
    const dataLines: string[] = [];
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) eventName = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    let p: any = {};
    const raw = dataLines.join('\n');
    if (raw) { try { p = JSON.parse(raw); } catch { p = { raw }; } }

    // agent_trace wraps the real event: type in p.type, payload in p.detail.
    if (eventName === 'agent_trace' && p && p.type) {
      const det = (p.detail && typeof p.detail === 'object') ? p.detail : {};
      eventName = p.type;
      p = { ...det, agent: p.agent, node: p.node };
    }

    // Harvest artifacts from ANY event, fire onFile with the deduped set.
    const before = downloads.size;
    collectDownloads(p, downloads);
    if (downloads.size !== before) h.onFile?.([...downloads.values()]);

    // Capture continuity ids from anywhere.
    const det = p.detail || p;
    const threadId = det?.thread_id || p.threadId;
    const memoryId = det?.memory_id || p.memoryId;
    if (threadId) state.threadId = threadId;
    if (memoryId) state.memoryId = memoryId;
    if (threadId || memoryId) h.onIds?.({ threadId: state.threadId, memoryId: state.memoryId });

    switch (eventName) {
      case 'tool_call': {
        const name = p.name || p.tool_calls?.[0]?.name || 'tool';
        const args = p.args !== undefined ? p.args : p.tool_calls?.[0]?.args;
        pending.add(name);
        batchWidth = Math.max(batchWidth, pending.size);
        h.onToolCall?.(name, parseMaybeJson(args) ?? args ?? {});
        break;
      }
      case 'tool_result': {
        const name = p.tool_name || p.name || 'tool';
        const rawContent = p.content !== undefined ? p.content : p.message;
        // The trace used to show that a tool was CALLED and never what came back, so a search
        // finding eight documents, one finding none, and one that failed all rendered
        // identically. The server now sends a headline and a duration; this is the line.
        //
        // The result row is indented under the call above it, which silently assumes the two
        // are adjacent. They are not when the model batches: one measured turn fired
        // keyword/semantic/spatial/opengeodata search and THEN printed four result rows, so
        // "2 results" could have belonged to any of them — and measured live, the results
        // come back OUT of call order, so the indent was not merely unproven but wrong. Name
        // the tool for every result in a batch; stay quiet when there was only one call and
        // the row above it is unambiguous.
        pending.delete(name);
        const batched = batchWidth > 1;
        if (pending.size === 0) batchWidth = 0;
        const bits = [batched ? `${name}: ${p.outcome ?? 'done'}` : p.outcome,
                      typeof p.duration_s === 'number' ? `${p.duration_s}s` : null]
          .filter(Boolean);
        if (bits.length) h.onTrace?.({ text: bits.join(' · '), kind: 'result' });
        h.onToolResult?.(name, parseMaybeJson(rawContent), rawContent);
        break;
      }
      // Tagged so the transcript can fold the ladder of these away at render while keeping
      // every one of them in the stored array. Untagged they arrived via the default branch
      // and were indistinguishable from any other status line.
      case 'llm_start': {
        // The server's line is "<model> started with 12 message(s)" — a count of the CONTEXT
        // window's history, sitting in a column where every other number is a result count,
        // and carrying the "(s)" hedge. The model name is the part worth stating once, and it
        // arrives on its own field, so the line is rebuilt here rather than reworded there.
        const msg = p.model ? `Asking ${p.model}` : (p.message || p.detail?.message);
        if (msg) h.onTrace?.({ text: String(msg), kind: 'llm' });
        break;
      }
      case 'tool_error':
        h.onTrace?.({ text: `${p.tool_name || p.name || 'tool'} failed: ${p.message || 'error'}`, kind: 'warn' });
        break;
      // The repair story. Without these the transcript shows a tool called twice and never says
      // the first attempt failed — the reader cannot tell a retry from a duplicate.
      case 'tool_retry':
      case 'tool_dead_end':
        if (p.message) h.onTrace?.({ text: String(p.message), kind: 'warn' });
        break;
      case 'tool_recovered':
        if (p.message) h.onTrace?.({ text: String(p.message), kind: 'recovered' });
        break;
      case 'llm_error':
        if (p.message) h.onTrace?.({ text: String(p.message), kind: 'warn' });
        break;
      // The peer's report to the supervisor, up to 4000 chars of raw markdown with URLs in
      // it — and the synthesised answer directly below the trace says the same thing. It was
      // the single largest row in every turn and the only one that duplicated the answer.
      // Dropped from the RENDER, not from the wire: it is still the only place to see what a
      // peer concluded before synthesis rewrote it, so other clients and anyone debugging
      // the stream keep it.
      case 'llm_message':
        break;
      case 'answer': {
        const t = p.final_answer || p.answer || p.detail?.final_answer || p.detail?.answer;
        if (t) { state.answer = t; h.onAnswerChunk?.(t); }
        break;
      }
      case 'map_layer': {
        const layer = (p.geojson || p.url ? p : p.detail) || {};
        if (layer.url && !layer.geojson) {
          h.onMapLayer?.({
            id: layer.id || 'agent-layer', source: layer.source || 'analysis',
            label: layer.label || 'Agent layer', count: layer.count,
            url: layer.url, render: layer.render, styleBy: layer.style_by ?? layer.styleBy,
            legend: _legend(layer), outline: layer.outline === true,
            sampled: !!layer.sampled, total: layer.total,
            // A raster layer is an image + its footprint; without bounds it cannot be placed.
            bounds: Array.isArray(layer.bounds) && layer.bounds.length === 4
              ? (layer.bounds.map(Number) as [number, number, number, number]) : undefined,
            opacity: typeof layer.opacity === 'number' ? layer.opacity : undefined,
            // Second of two field-by-field rebuilds between the tool and the layer list (the
            // other is the server's build_map_layer). A field missing from EITHER is gone, so
            // the pointer to the layer's vectors has to be named in both.
            embedding: layer.embedding && layer.embedding.file_id ? layer.embedding : undefined,
          });
        } else if (layer.geojson && Array.isArray(layer.geojson.features)) {
          h.onMapLayer?.({
            id: layer.id || 'agent-layer',
            source: layer.source || 'analysis',
            label: layer.label || 'Agent layer',
            count: layer.count,
            geojson: layer.geojson,
            render: layer.render, styleBy: layer.style_by ?? layer.styleBy,
            legend: _legend(layer), outline: layer.outline === true,
            sampled: !!layer.sampled, total: layer.total,
          });
        }
        break;
      }
      case 'response':
      case 'result':
        state.response = p;
        if (!state.answer && (p.answer || p.final_answer)) state.answer = p.answer || p.final_answer;
        break;
      case 'error':
        state.error = p.error || p.message || 'Request failed';
        break;
      // The graph's own progress: "Routing the request", "Running analysis workflow",
      // "supervisor -> analyze (decision)", "Composing answer". There was no case for it, so
      // every one of these fell through to `default:` and was pushed with NO kind — which is
      // also where the peer's raw prose lands. Indistinguishable rows cannot be folded, which
      // is why nine of the fifteen rows in a one-tool turn were the framework announcing
      // itself. Tagging them is what lets the transcript collapse the ladder.
      // Why the loop did something a reader would otherwise call a bug — stopped early,
      // stopped without searching, ran analysis on a request that named none. Its own kind
      // so the routing fold cannot swallow it.
      case 'decision': {
        if (p.message) h.onTrace?.({ text: String(p.message), kind: 'decision' });
        break;
      }
      case 'node': {
        const msg = p.message || p.detail?.message;
        if (msg) h.onTrace?.({ text: String(msg), kind: 'node' });
        break;
      }
      case 'status':
      case 'routing':
      // Emitted (via _category_for_agent_role) but carrying tool payloads whose message sits
      // under .detail, so these render nothing today. Kept as explicit no-ops rather than
      // deleted: they ARE on the wire, and a future payload with a top-level message belongs
      // here rather than in `default:`.
      case 'search':
      case 'analysis': {
        const msg = p.message || p.label || p.stage || p.route;
        if (msg) h.onTrace?.({ text: String(msg), kind: eventName });
        break;
      }
      default: {
        const msg = p.message || p.content;
        if (msg && typeof msg === 'string') h.onTrace?.({ text: msg });
      }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx = buf.indexOf('\n\n');
    while (idx !== -1) { handleBlock(buf.slice(0, idx)); buf = buf.slice(idx + 2); idx = buf.indexOf('\n\n'); }
  }
  if (buf.trim()) handleBlock(buf);

  state.downloads = [...downloads.values()];
  return state;
}

// Best-effort geometry extraction: scan a payload for anything plottable so the map
// can show KB/spatial results if the backend surfaces coordinates (defensive: the
// current contract usually does NOT include element geometry -- see agent-api-contract).
import type { Feature, FeatureCollection, Geometry } from 'geojson';

/** Class-name -> swatch pairs for a categorical layer, dropping malformed entries.
 *  A categorical layer is only categorical if its legend survives the wire: App keys the
 *  'categories' render off `legend?.length`, so an omitted legend silently becomes a flat fill. */
function _legend(layer: any): { label: string; color: [number, number, number, number] }[] | undefined {
  const raw = layer?.legend;
  if (!Array.isArray(raw)) return undefined;
  const out = raw
    .filter((e: any) => e && typeof e.label === 'string' && Array.isArray(e.color) && e.color.length >= 3)
    .map((e: any) => ({
      label: String(e.label),
      color: [Number(e.color[0]), Number(e.color[1]), Number(e.color[2]),
              e.color.length > 3 ? Number(e.color[3]) : 255] as [number, number, number, number],
    }));
  return out.length ? out : undefined;
}

export function extractFeatures(payload: any): FeatureCollection {
  const feats: Feature[] = [];
  const seen = new Set<any>();
  const visit = (v: any) => {
    if (!v || typeof v !== 'object' || seen.has(v)) return;
    seen.add(v);
    if (Array.isArray(v)) { v.forEach(visit); return; }
    // Direct GeoJSON geometry container
    const geom = pickGeometry(v);
    if (geom) {
      feats.push({ type: 'Feature', geometry: geom, properties: { name: v.title || v.name || v.doc_id || v.id || '(item)', ...flatProps(v) } });
    }
    Object.values(v).forEach(visit);
  };
  visit(payload);
  return { type: 'FeatureCollection', features: feats };
}

function pickGeometry(v: any): Geometry | null {
  const g = v['spatial-geometry'] || v['spatial-bounding-box'] || v.geometry || v.geom;
  if (g && typeof g === 'object' && g.type && g.coordinates) return g as Geometry;
  const c = v['spatial-centroid'] || v.centroid;
  if (c && c.type === 'Point' && Array.isArray(c.coordinates)) return c as Geometry;
  const lat = v.lat ?? v.latitude, lon = v.lon ?? v.lng ?? v.longitude;
  if (typeof lat === 'number' && typeof lon === 'number') return { type: 'Point', coordinates: [lon, lat] };
  return null;
}

function flatProps(v: any): Record<string, any> {
  const out: Record<string, any> = {};
  for (const k of ['title', 'resource-type', 'source', 'doc_id', 'id', 'score']) if (v[k] != null) out[k] = v[k];
  return out;
}
