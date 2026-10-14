// Replay a recorded turn through the real client. The GIS harness keeps every SSE event of a
// turn as `<task>.events.jsonl`, one `{"event": name, "data": payload}` per line, which is the
// SSE block the server sent with the framing taken off. Put the framing back and the stream can
// go through `streamChat` unchanged: the same parser, the same `collectDownloads`, the same
// `map_layer` drawing. That is the point. A re-implementation of what the UI would show is a
// second opinion; the UI showing it is the evidence (AGENTS.md "Verifying UI and delivery
// changes").
//
// Pure functions only, shared by the replay server (scripts/replay-server.ts), the replay strip
// in the page, and `npm run check:replay`.

export interface RecordedEvent { event: string; data: any }

/** One `map_layer` event, reduced to what decides whether the UI can draw it. */
export interface ReplayLayer {
  n: number;                 // 1-based, in stream order: the NN of layers/<turn>__NN__<label>.*
  id: string;
  label: string;
  render?: string;
  fileId?: string;           // from the url; the file the UI will ask /agent/files/<id>/download for
  url?: string;
  inline: boolean;           // geojson carried in the event itself
  /** Why the UI cannot draw this layer whatever the replay serves, or null when it can. */
  problem: string | null;
}

/** What the server knows about a replayed turn: sent to the page as /agent/replay/<key>. */
export interface ReplayMeta {
  key: string;
  title: string;             // run / model / turn, for the strip
  query: string;
  queryFrom: 'record' | 'route_trace' | 'none';
  events: number;
  badLines: number;
  /** What normaliseRecording undid: hosts made host-relative, layers' JSON text parsed back. */
  rewrittenHosts: string[];
  geojsonParsed: number;
  answerFrom: 'stream' | 'record' | 'none';
  layers: (ReplayLayer & { served: 'inline' | 'file' | 'missing' | 'n/a' })[];
}

export function parseJsonl(text: string): { events: RecordedEvent[]; bad: number } {
  const events: RecordedEvent[] = [];
  let bad = 0;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const e = JSON.parse(line);
      if (e && typeof e.event === 'string') events.push({ event: e.event, data: e.data });
      else bad++;
    } catch { bad++; }
  }
  return { events, bad };
}

/** The SSE body the server sent. `JSON.stringify` never emits a raw newline, so every payload
 *  is one `data:` line and a blank line ends each block, as streamChat splits them. */
export function toSse(events: RecordedEvent[]): string {
  return events.map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data ?? {})}\n\n`).join('');
}

const FILE_ID = /\/agent\/files\/([^/?#]+)\/download/;
const ABSOLUTE_FILE_URL = /https?:\/\/([^\s"'/\\]+)(\/agent\/files\/[^/?#\s"'\\]+\/download)/g;

/** Undo what recording did to a turn, and nothing else. Two changes, both counted for the strip:
 *
 *  - Every agent-file url becomes host-relative, so the replay serves it from the capture. A
 *    turn recorded against a deployment carries absolute urls (https://agent.i-guide.io/agent/
 *    files/<id>/download). Left alone, the replayed page asks that server for each layer: a
 *    request to production for a file the run already captured, refused anyway without a
 *    sign-in, so the layer would be reported missing when it is not.
 *  - A map_layer's geojson stored as JSON TEXT is parsed back. chat_traces keeps event payloads
 *    as text (the index mapping, PRs #86 and #92), so a live round pulled from it has every
 *    inline layer as a string, where the stream had sent an object. A string that is not JSON
 *    (a Python repr) is left alone and reported: that one may have been what was sent. */
export function normaliseRecording(events: RecordedEvent[]):
    { events: RecordedEvent[]; hosts: string[]; geojsonParsed: number } {
  const hosts = new Set<string>();
  let geojsonParsed = 0;
  const out = events.map((e) => {
    let data = e.data;
    const text = JSON.stringify(data ?? {});
    if (text.includes('/agent/files/')) {
      const local = text.replace(ABSOLUTE_FILE_URL, (_m, host: string, path: string) => { hosts.add(host); return path; });
      if (local !== text) data = JSON.parse(local);
    }
    if (e.event === 'map_layer' && data && typeof data.geojson === 'string' && /^\s*\{\s*"/.test(data.geojson)) {
      try { data = { ...data, geojson: JSON.parse(data.geojson) }; geojsonParsed++; } catch { /* reported as a string */ }
    }
    return data === e.data ? e : { event: e.event, data };
  });
  return { events: out, hosts: [...hosts].sort(), geojsonParsed };
}

/** The `map_layer` events, as streamChat will see them. Mirrors its acceptance exactly: an
 *  `event: map_layer` block, or an `agent_trace` whose type is map_layer; the descriptor is the
 *  payload itself when it carries geojson or a url, else its `detail`. */
export function replayLayers(events: RecordedEvent[]): ReplayLayer[] {
  const out: ReplayLayer[] = [];
  for (const e of events) {
    let name = e.event;
    let p: any = e.data && typeof e.data === 'object' ? e.data : {};
    if (name === 'agent_trace' && p.type) {
      const det = p.detail && typeof p.detail === 'object' ? p.detail : {};
      name = p.type;
      p = { ...det, agent: p.agent, node: p.node };
    }
    if (name !== 'map_layer') continue;
    const layer = (p.geojson || p.url ? p : p.detail) || {};
    const n = out.length + 1;
    const label = String(layer.label || 'Agent layer');
    const url = typeof layer.url === 'string' ? layer.url : undefined;
    const fileId = url ? FILE_ID.exec(url)?.[1] : undefined;
    const inline = !url && !!layer.geojson;
    let problem: string | null = null;
    if (url) {
      if (layer.render === 'raster') {
        const b = layer.bounds;
        if (!Array.isArray(b) || b.length !== 4) problem = 'raster without bounds, so it cannot be placed';
        else if (!boundsAreLonLat(b.map(Number))) {
          problem = `raster bounds are not lon/lat (${b.map((v: any) => Math.round(Number(v))).join(', ')}), so it lands off the map`;
        }
      }
    } else if (layer.geojson) {
      if (!Array.isArray(layer.geojson.features)) {
        problem = typeof layer.geojson === 'string'
          ? 'inline geojson arrived as a string, which the client does not parse'
          : 'inline geojson has no features array';
      } else if (!layer.geojson.features.length) problem = 'inline geojson has no features';
    } else problem = 'event has neither a url nor inline geojson';
    out.push({ n, id: String(layer.id || 'agent-layer'), label, render: layer.render, fileId, url,
               inline, problem });
  }
  return out;
}

export function boundsAreLonLat(b: number[]): boolean {
  return b.length === 4 && b.every(Number.isFinite)
    && Math.abs(b[0]) <= 180 && Math.abs(b[2]) <= 180 && Math.abs(b[1]) <= 90 && Math.abs(b[3]) <= 90;
}

/** The question the user typed. The harness record has it; a turn recorded without one (a live
 *  case driven by a script) may still carry it in its route trace. */
export function queryOf(record: any, events: RecordedEvent[]): { query: string; from: ReplayMeta['queryFrom'] } {
  if (record && typeof record.query === 'string' && record.query.trim()) return { query: record.query, from: 'record' };
  for (const e of events) {
    const d = e.data;
    const q = d?.query ?? d?.detail?.query;
    if (typeof q === 'string' && q.trim()) return { query: q, from: 'route_trace' };
  }
  return { query: '', from: 'none' };
}

/** The answer, when the recording lacks it. A live round pulled from chat_traces keeps the
 *  trace events but not the `answer`/`result` ones, and the answer is in the record instead. It
 *  is appended as the terminal `result` the server would have sent, so the page renders it the
 *  same way, and the strip says where it came from. */
export function withRecordedAnswer(events: RecordedEvent[], record: any):
    { events: RecordedEvent[]; answerFrom: 'stream' | 'record' | 'none' } {
  if (events.some((e) => ['answer', 'result', 'response'].includes(e.event))) return { events, answerFrom: 'stream' };
  const a = record && typeof record.answer === 'string' && record.answer.trim() ? record.answer : null;
  return a ? { events: [...events, { event: 'result', data: { answer: a } }], answerFrom: 'record' }
           : { events, answerFrom: 'none' };
}

/** One line per layer the page could not draw, and the reason, for the strip on the
 *  screenshot. `drawn` is the set of layer ids the page actually holds once the turn ended. */
export function layerVerdicts(meta: Pick<ReplayMeta, 'layers'>, drawn: Set<string>):
    { drawn: number; total: number; distinct: number; lines: string[] } {
  const lines: string[] = [];
  let ok = 0;
  for (const l of meta.layers) {
    if (drawn.has(l.id)) {
      ok++;
      if (l.problem) lines.push(`#${l.n} ${l.label}: on the layer list, but ${l.problem}`);
      continue;
    }
    const why = l.problem
      ?? (l.served === 'missing' ? `its file ${l.fileId ?? '(no file id)'} was not captured with the run`
          : 'the page did not draw it (empty or unreadable file)');
    lines.push(`#${l.n} ${l.label}: NOT drawn: ${why}`);
  }
  return { drawn: ok, total: meta.layers.length, distinct: new Set(meta.layers.map((l) => l.id)).size, lines };
}
