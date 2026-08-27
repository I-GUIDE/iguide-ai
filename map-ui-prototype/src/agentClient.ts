// Real client for the I-GUIDE agent SSE API (see api/server.py :: /agent/chat/stream
// and examples/iguide_chat_prototype.html). Event-driven: per-event hooks fire live
// so the map/chat update AS the agent works; onFinal reconciles against the
// authoritative terminal `result`. This module IS the swap point that replaces the
// local deterministic agentBrain.
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
}

export interface ModelCatalogue {
  default: { provider: string; model: string };
  /** Accepted reasoning_effort values, per the API's own error message. */
  reasoning_efforts?: string[];
  providers: { provider: string; label: string; configured: boolean;
               models: string[]; stale?: boolean;
               /** Subset of `models` that accept reasoning_effort. */
               reasoning_models?: string[];
               /** Legal reasoning_effort values PER model, with function tools attached.
                *  A global list offered 'high' on models that refuse any real level. */
               effort_options?: Record<string, string[]>;
               /** Models that REFUSE tools unless this exact value is sent (gpt-5.6-*). */
               effort_required?: Record<string, string> }[];
}

/** Ask the agent which models a request may select. */
export async function fetchModels(cfg: AgentConfig): Promise<ModelCatalogue | null> {
  try {
    // apiBase is only the ORIGIN, so name the path explicitly — the same way download
    // URLs are resolved. new URL('models', origin) would hit /models.
    const url = absoluteUrl('/agent/models', cfg);
    const r = await fetch(url, { headers: authHeaders(cfg, false) });
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
  styleBy?: string;             // property to shade by (numeric, or a class name for 'categories')
  legend?: { label: string; color: [number, number, number, number] }[];
  style_by?: string;            // snake_case as the agent sends it
  sampled?: boolean;            // true when the layer is a subset of the data
  total?: number;               // full population size when sampled
  bounds?: [number, number, number, number];  // raster footprint [minLon,minLat,maxLon,maxLat]
  opacity?: number;             // raster draping opacity
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
  const res = await fetch(cfg.uploadEndpoint, { method: 'POST', headers: authHeaders(cfg, false), body: fd });
  if (!res.ok) throw new Error(await describeError(res));
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
  };

  const resp = await fetch(cfg.endpoint, {
    method: 'POST', headers: authHeaders(cfg, true),
    body: JSON.stringify(payload), signal: opts.signal,
  });
  if (!resp.ok || !resp.body) throw new Error(await describeError(resp));

  const downloads = new Map<string, FileRecord>();
  const state: StreamResult = { answer: '', response: null, downloads: [], threadId: opts.threadId, memoryId: opts.memoryId ?? undefined };
  const reader = resp.body.getReader();
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
        h.onToolCall?.(name, parseMaybeJson(args) ?? args ?? {});
        break;
      }
      case 'tool_result': {
        const name = p.tool_name || p.name || 'tool';
        const rawContent = p.content !== undefined ? p.content : p.message;
        h.onToolResult?.(name, parseMaybeJson(rawContent), rawContent);
        break;
      }
      case 'tool_error':
        h.onTrace?.({ text: `${p.tool_name || p.name || 'tool'} failed: ${p.message || 'error'}`, kind: 'warn' });
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
            legend: _legend(layer),
            sampled: !!layer.sampled, total: layer.total,
            // A raster layer is an image + its footprint; without bounds it cannot be placed.
            bounds: Array.isArray(layer.bounds) && layer.bounds.length === 4
              ? (layer.bounds.map(Number) as [number, number, number, number]) : undefined,
            opacity: typeof layer.opacity === 'number' ? layer.opacity : undefined,
          });
        } else if (layer.geojson && Array.isArray(layer.geojson.features)) {
          h.onMapLayer?.({
            id: layer.id || 'agent-layer',
            source: layer.source || 'analysis',
            label: layer.label || 'Agent layer',
            count: layer.count,
            geojson: layer.geojson,
            render: layer.render, styleBy: layer.style_by ?? layer.styleBy,
            legend: _legend(layer),
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
      case 'status':
      case 'routing':
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
