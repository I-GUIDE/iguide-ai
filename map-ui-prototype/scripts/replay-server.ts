// The agent API, played back from a recording. Serves just enough of /agent/* for the map UI to
// run a recorded turn as if it were live:
//
//   GET  /agent/replay/meta?key=K        what the page needs before it sends: the query, the layers
//   POST /agent/chat/stream?replay=K     the recorded events, re-framed as the SSE the server sent
//   GET  /agent/files/<file_id>/download a layer file captured with the run
//   anything else under /agent/          404, which the UI already treats as "no such feature"
//
// Layer files: the run's map_layer urls point at a local server that no longer exists. The harness
// captures each layer at run time, and tools/dump_layers.py did the same for the older runs, both
// as <dir>/layers/<turn>__NN__<label>.<ext>, NN being the layer's place among the turn's
// map_layer events. A file that was never captured gets a 404 with a reason, which the UI already
// reports, and the replay strip names on the screenshot. Nothing is drawn in its place.
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { basename, dirname, extname, join } from 'node:path';
import { normaliseRecording, parseJsonl, queryOf, replayLayers, toSse, withRecordedAnswer,
  type ReplayMeta } from '../src/replay';

export interface ReplayTurn { meta: ReplayMeta; sse: string }

const TYPES: Record<string, string> = {
  '.geojson': 'application/geo+json', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.csv': 'text/csv', '.tif': 'image/tiff',
  '.tiff': 'image/tiff', '.svg': 'image/svg+xml', '.html': 'text/html', '.txt': 'text/plain',
};

export function stemOf(eventsPath: string): string {
  return basename(eventsPath).replace(/\.events\.jsonl$/, '');
}

/** Read one recorded turn. `files` collects file_id -> path for every layer found on disk. */
export function loadTurn(key: string, eventsPath: string, files: Map<string, string>): ReplayTurn {
  const dir = dirname(eventsPath);
  const stem = stemOf(eventsPath);
  const parsed = parseJsonl(readFileSync(eventsPath, 'utf8'));
  const norm = normaliseRecording(parsed.events);
  const { hosts, geojsonParsed } = norm;
  const bad = parsed.bad;
  const recPath = join(dir, `${stem}.json`);
  let record: any = null;
  try { record = existsSync(recPath) ? JSON.parse(readFileSync(recPath, 'utf8')) : null; } catch { /* */ }
  const { events, answerFrom } = withRecordedAnswer(norm.events, record);
  const { query, from } = queryOf(record, events);
  const layerDir = join(dir, 'layers');
  const dumped = existsSync(layerDir) ? readdirSync(layerDir) : [];
  const serverFiles = join(dir, '..', 'server_files');
  const layers = replayLayers(events).map((l) => {
    if (l.inline) return { ...l, served: 'inline' as const };
    if (!l.fileId) return { ...l, served: 'n/a' as const };
    const prefix = `${stem}__${String(l.n).padStart(2, '0')}__`;
    const hit = dumped.find((f) => f.startsWith(prefix) && !f.endsWith('.bounds.json'));
    const path = hit ? join(layerDir, hit) : findServerFile(serverFiles, l.fileId);
    if (path) files.set(l.fileId, path);
    return { ...l, served: path ? 'file' as const : 'missing' as const };
  });
  // Downloads and inline images are fetched by file_id too; serve any the run kept.
  for (const m of readFileSync(eventsPath, 'utf8').matchAll(/\/agent\/files\/(file_[0-9a-f]+)\/download/g)) {
    if (!files.has(m[1])) { const p = findServerFile(serverFiles, m[1]); if (p) files.set(m[1], p); }
  }
  const parts = eventsPath.split(/[\\/]/);
  const title = parts.slice(-3, -1).concat(stem).join(' / ');
  return { meta: { key, title, query, queryFrom: from, events: events.length, badLines: bad,
                   rewrittenHosts: hosts, geojsonParsed, answerFrom, layers },
           sse: toSse(events) };
}

function findServerFile(root: string, fileId: string): string | null {
  for (const sub of ['outputs', 'uploads']) {
    const d = join(root, sub);
    if (!existsSync(d)) continue;
    const f = readdirSync(d).find((n) => n.startsWith(fileId) && !n.endsWith('.aux.xml'));
    if (f) return join(d, f);
  }
  return null;
}

/** A connect-style handler. `resolve` maps a key from the page to an events file, or null; it is
 *  the only way in, so a key can never name a path the caller did not offer. Returns false when
 *  the request is not the replay's to answer. */
export function replayHandler(resolve: (key: string) => string | null) {
  const turns = new Map<string, ReplayTurn>();
  const files = new Map<string, string>();
  const turn = (key: string | null): ReplayTurn | null => {
    if (!key) return null;
    if (!turns.has(key)) {
      const p = resolve(key);
      if (!p || !existsSync(p)) return null;
      turns.set(key, loadTurn(key, p, files));
    }
    return turns.get(key)!;
  };
  const json = (res: ServerResponse, status: number, body: unknown) => {
    res.writeHead(status, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    res.end(JSON.stringify(body));
  };
  return (req: IncomingMessage, res: ServerResponse): boolean => {
    const url = new URL(req.url || '/', 'http://replay');
    if (!url.pathname.startsWith('/agent/')) return false;
    if (url.pathname === '/agent/replay/meta') {
      const t = turn(url.searchParams.get('key'));
      t ? json(res, 200, t.meta) : json(res, 404, { reason: 'no such recorded turn' });
      return true;
    }
    if (url.pathname === '/agent/chat/stream') {
      const t = turn(url.searchParams.get('replay'));
      if (!t) { json(res, 404, { reason: 'no such recorded turn' }); return true; }
      req.resume();                                   // the request body is the page's; unused
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      res.end(t.sse);
      return true;
    }
    const m = /^\/agent\/files\/([^/]+)\/download$/.exec(url.pathname);
    if (m) {
      const p = files.get(m[1]);
      if (!p) { json(res, 404, { reason: 'not captured with the run' }); return true; }
      res.writeHead(200, { 'content-type': TYPES[extname(p).toLowerCase()] || 'application/octet-stream',
                           'cache-control': 'no-store' });
      res.end(readFileSync(p));
      return true;
    }
    json(res, 404, { reason: 'not part of a replay' });
    return true;
  };
}
