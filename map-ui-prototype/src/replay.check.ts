/** A runnable check of the replay reducer and server. `npm run check:replay`.
 *
 * Same reasoning as the other check: scripts (there is no test runner here). The claim of a
 * replay screenshot is that it shows what the UI would have shown, which holds only while the
 * recording goes through the client's OWN parser. So the core of this check pipes a recorded
 * turn through the replay server and into the real `streamChat`, and compares what its handlers
 * saw with the events that were recorded.
 */
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable } from 'node:stream';
import { streamChat, type AgentConfig, type FileRecord, type MapLayerEvent } from './agentClient';
import { layerVerdicts, normaliseRecording, parseJsonl, queryOf, replayLayers, toSse, withRecordedAnswer,
  type RecordedEvent } from './replay';
import { replayHandler } from '../scripts/replay-server';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}${ok ? '' : `, want ${JSON.stringify(want)}`}`);
};

const fc = (n: number) => ({ type: 'FeatureCollection',
  features: Array.from({ length: n }, (_, i) => ({ type: 'Feature', properties: { i },
    geometry: { type: 'Point', coordinates: [-87.7 + i / 100, 41.9] } })) });

// A turn as the harness records it: { event, data } per line, in stream order.
const recorded: RecordedEvent[] = [
  { event: 'status', data: { status: 'Agent chat started' } },
  { event: 'routing', data: { type: 'route_trace', detail: { query: 'schools near the point?' } } },
  { event: 'agent_trace', data: { type: 'tool_call', detail: { name: 'overpass_search', args: { tag: 'amenity=school' } } } },
  // Two blank lines INSIDE a payload: the SSE framing must survive them.
  { event: 'agent_trace', data: { type: 'tool_result', detail: { tool_name: 'overpass_search', outcome: '23 features',
    content: 'line one\n\nline two', download_url: '/agent/files/file_aaaaaaaaaaaa/download',
    file_id: 'file_aaaaaaaaaaaa', filename: 'osm_school.geojson' } } },
  { event: 'map_layer', data: { kind: 'map_layer', id: 'L1', label: 'OSM: school', geojson: fc(3), count: 3 } },
  { event: 'map_layer', data: { kind: 'map_layer', id: 'L2', label: 'Within 1 mile', render: 'points',
    url: '/agent/files/file_bbbbbbbbbbbb/download', count: 2 } },
  { event: 'map_layer', data: { kind: 'map_layer', id: 'L3', label: 'Never captured', render: 'shapes',
    url: '/agent/files/file_cccccccccccc/download', count: 1 } },
  { event: 'map_layer', data: { kind: 'map_layer', id: 'L4', label: 'Slope', render: 'raster',
    url: '/agent/files/file_dddddddddddd/download', bounds: [390000, 4434000, 394500, 4440000] } },
  // The wrapped form streamChat also accepts: the descriptor sits under agent_trace.detail.
  { event: 'agent_trace', data: { type: 'map_layer', detail: { id: 'L5', label: 'As a string',
    geojson: "{'type': 'FeatureCollection', 'features': []}" } } },
  { event: 'answer', data: { type: 'completed', final_answer: '**20 schools.**\n\n⚠️ Check this answer.' } },
  { event: 'result', data: { answer: '**20 schools.**\n\n⚠️ Check this answer.' } },
];

// --- the reducer -------------------------------------------------------------------------
const jsonl = recorded.map((e) => JSON.stringify(e)).join('\n') + '\n\nnot json\n{"no_event": 1}\n';
const parsed = parseJsonl(jsonl);
eq('parseJsonl keeps every recorded event', parsed.events.length, recorded.length);
eq('  ...and counts the lines it could not read', parsed.bad, 2);

const sse = toSse(parsed.events);
eq('toSse writes one block per event', sse.split('\n\n').filter(Boolean).length, recorded.length);
eq('  ...each payload on a single data: line', sse.split('\n').filter((l) => l.startsWith('data:')).length, recorded.length);

const layers = replayLayers(parsed.events);
eq('replayLayers finds every map_layer, wrapped or not', layers.map((l) => l.id), ['L1', 'L2', 'L3', 'L4', 'L5']);
eq('  ...numbered in stream order, as layers/<turn>__NN__ is', layers.map((l) => l.n), [1, 2, 3, 4, 5]);
eq('  ...with the file id the page will ask for', layers.map((l) => l.fileId ?? null),
   [null, 'file_bbbbbbbbbbbb', 'file_cccccccccccc', 'file_dddddddddddd', null]);
eq('  ...a projected raster is named as unplaceable', /not lon\/lat/.test(layers[3].problem || ''), true);
eq('  ...string geojson is named as undrawable', /as a string/.test(layers[4].problem || ''), true);
eq('  ...the drawable ones carry no problem', layers.slice(0, 3).map((l) => l.problem), [null, null, null]);

eq('queryOf prefers the harness record', queryOf({ query: 'typed' }, parsed.events), { query: 'typed', from: 'record' });
eq('  ...then the route trace', queryOf(null, parsed.events), { query: 'schools near the point?', from: 'route_trace' });
eq('  ...and says when there is none', queryOf({}, []), { query: '', from: 'none' });

// A turn recorded against a deployment: its urls are absolute and must not be fetched from there.
const deployed: RecordedEvent[] = [
  { event: 'map_layer', data: { id: 'D1', label: 'County', url: 'https://agent.i-guide.io/agent/files/file_eeeeeeeeeeee/download' } },
  { event: 'result', data: { files: [{ file_id: 'file_ffffffffffff', filename: 'x.png',
    download_url: 'http://localhost:8000/agent/files/file_ffffffffffff/download' }],
    answer: 'see https://agent.i-guide.io/about and [x](https://agent.i-guide.io/agent/files/file_ffffffffffff/download)' } },
];
const loc = normaliseRecording(deployed);
eq('normaliseRecording makes layer urls host-relative', loc.events[0].data.url, '/agent/files/file_eeeeeeeeeeee/download');
eq('  ...and download urls, in nested records and answer text',
   [loc.events[1].data.files[0].download_url, loc.events[1].data.answer],
   ['/agent/files/file_ffffffffffff/download', 'see https://agent.i-guide.io/about and [x](/agent/files/file_ffffffffffff/download)']);
eq('  ...names the hosts it rewrote', loc.hosts, ['agent.i-guide.io', 'localhost:8000']);
eq('  ...and leaves a host-relative recording as it is',
   normaliseRecording(recorded).events.every((e, i) => e === recorded[i]), true);
// chat_traces stores payloads as text: JSON text is parsed back, a Python repr is not.
const stored = normaliseRecording([
  { event: 'map_layer', data: { id: 'S1', label: 'Stored', geojson: JSON.stringify(fc(2)) } },
  { event: 'map_layer', data: { id: 'S2', label: 'Repr', geojson: "{'type': 'FeatureCollection'}" } },
]);
eq('  ...parses geojson stored as JSON text', [stored.geojsonParsed, stored.events[0].data.geojson.features.length], [1, 2]);
eq('  ...and leaves a repr string to be reported', typeof stored.events[1].data.geojson, 'string');
eq('  ...so only the repr is a problem', replayLayers(stored.events).map((l) => l.problem === null), [true, false]);

// A live round pulled from chat_traces has no answer event; its record does.
const traceOnly = recorded.filter((e) => !['answer', 'result'].includes(e.event));
eq('withRecordedAnswer keeps a streamed answer', withRecordedAnswer(recorded, { answer: 'other' }).answerFrom, 'stream');
const fromRec = withRecordedAnswer(traceOnly, { answer: 'Mean slope **0.47°**.' });
eq('  ...else appends the record\'s as the terminal result',
   [fromRec.answerFrom, fromRec.events[fromRec.events.length - 1]], ['record', { event: 'result', data: { answer: 'Mean slope **0.47°**.' } }]);
eq('  ...and says when there is none', withRecordedAnswer(traceOnly, {}).answerFrom, 'none');

// --- the server, on a run laid out as the harness writes it ---------------------------------
const dir = mkdtempSync(join(tmpdir(), 'replaycheck-'));
const turnDir = join(dir, 'run', 'lumen_model');
mkdirSync(join(turnDir, 'layers'), { recursive: true });
writeFileSync(join(turnDir, 'T02L.events.jsonl'), jsonl);
writeFileSync(join(turnDir, 'T02L.json'), JSON.stringify({ query: 'How many schools within 1 mile?' }));
writeFileSync(join(turnDir, 'layers', 'T02L__01__OSM_school.geojson'), JSON.stringify(fc(3)));
writeFileSync(join(turnDir, 'layers', 'T02L__02__Within_1_mile.geojson'), JSON.stringify(fc(2)));
// T02L__03__ was never captured. A different turn's layer 3 must not stand in for it.
writeFileSync(join(turnDir, 'layers', 'T02L.t1__03__Never_captured.geojson'), JSON.stringify(fc(9)));
writeFileSync(join(turnDir, 'layers', 'T02L__04__Slope.png'), Buffer.from([0x89, 0x50, 0x4e, 0x47]));
writeFileSync(join(turnDir, 'layers', 'T02L__04__Slope.bounds.json'), '{}');

const handle = replayHandler((k) => (k === '0' ? join(turnDir, 'T02L.events.jsonl') : null));
async function call(method: string, url: string): Promise<{ status: number; type: string; body: Buffer }> {
  const req = Object.assign(Readable.from([]), { method, url }) as unknown as IncomingMessage;
  return new Promise((resolve) => {
    let status = 0; let type = '';
    const res = {
      writeHead(s: number, h: Record<string, string>) { status = s; type = h['content-type'] || ''; return res; },
      end(b?: string | Buffer) { resolve({ status, type, body: Buffer.from(b ?? '') }); },
    } as unknown as ServerResponse;
    if (!handle(req, res)) resolve({ status: -1, type: '', body: Buffer.from('') });
  });
}

const meta = JSON.parse((await call('GET', '/agent/replay/meta?key=0')).body.toString());
eq('meta: the query comes from the record', [meta.query, meta.queryFrom], ['How many schools within 1 mile?', 'record']);
eq('meta: title is run / model / turn', meta.title, 'run / lumen_model / T02L');
eq('meta: the answer is the stream\'s own', meta.answerFrom, 'stream');
eq('meta: what is served for each layer', meta.layers.map((l: any) => l.served), ['inline', 'file', 'missing', 'file', 'inline']);
eq('an unknown key is a 404', (await call('GET', '/agent/replay/meta?key=../../etc')).status, 404);
eq('a path outside /agent/ is not the handler\'s', (await call('GET', '/index.html')).status, -1);
const missing = await call('GET', '/agent/files/file_cccccccccccc/download');
eq('an uncaptured layer is a 404 with a reason', [missing.status, JSON.parse(missing.body.toString()).reason],
   [404, 'not captured with the run']);
const png = await call('GET', '/agent/files/file_dddddddddddd/download');
eq('a raster layer is served as an image', [png.status, png.type], [200, 'image/png']);

// --- end to end: the recording, through the server, through the REAL client -----------------
(globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown) => {
  const r = await call('POST', String(url));
  return new Response(new Uint8Array(r.body), { status: r.status, headers: { 'content-type': r.type } });
};
const cfg = { endpoint: '/agent/chat/stream?replay=0', uploadEndpoint: '/agent/files/upload', apiKey: '' } as AgentConfig;
const seen: MapLayerEvent[] = [];
let files: FileRecord[] = [];
const tools: string[] = [];
const res = await streamChat('How many schools within 1 mile?', { threadId: 't' }, cfg, {
  onMapLayer: (l) => seen.push(l), onFile: (f) => { files = f; }, onToolCall: (n) => tools.push(n),
});
eq('streamChat: the answer arrives', res.answer, '**20 schools.**\n\n⚠️ Check this answer.');
eq('streamChat: the tool call arrives', tools, ['overpass_search']);
eq('streamChat: downloads are collected from the recording', files.map((f) => f.filename), ['osm_school.geojson']);
// L5 is dropped by the client itself (its geojson is a string), exactly as it was live.
eq('streamChat: map layers, as the live client took them', seen.map((l) => l.id), ['L1', 'L2', 'L3', 'L4']);
eq('  ...inline geometry intact', seen[0].geojson?.features.length, 3);
eq('  ...urls untouched, so the page asks the replay for them', seen[1].url, '/agent/files/file_bbbbbbbbbbbb/download');

// --- the verdict the strip shows ------------------------------------------------------------
const v = layerVerdicts(meta, new Set(['L1', 'L2', 'L4']));
eq('verdict counts what the page drew', [v.drawn, v.total, v.distinct], [3, 5, 5]);
eq('  ...names the uncaptured file', v.lines.some((l) => /#3 Never captured: NOT drawn: its file file_cccccccccccc was not captured/.test(l)), true);
eq('  ...and a drawn layer that cannot be seen', v.lines.some((l) => /#4 Slope: on the layer list, but raster bounds/.test(l)), true);
eq('  ...and the string geojson', v.lines.some((l) => /#5 As a string: NOT drawn: inline geojson arrived as a string/.test(l)), true);
eq('  ...and nothing about the layers that drew', v.lines.length, 3);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
