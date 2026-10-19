/** A runnable check that an agent file fetched after the access cookie aged out refreshes the
 * sign-in once and is fetched once more. `npm run check:refresh`.
 *
 * Same reasoning as check:auth and check:files: no test runner here, and this fails in a way no
 * screenshot explains. The platform's access cookie lives one hour. On agent.i-guide.io on
 * 2026-10-18 a turn's chat call went through at 2 s, both of its layer downloads were refused
 * 401 `token_expired` at 60 s and 75 s, and the map stayed empty — while the platform's refresh
 * endpoint, called later for whoami, answered 200. fetchAgentFile had no retry at all.
 */
import { readFileSync } from 'node:fs';
import { cancelTurn, fetchAgentFile, fetchModels, type AgentConfig } from './agentClient';
import { setRefreshUrl } from './auth';
import { loadLayerData } from './layerLoad';
import { loadRasterImage, toDeckLayer } from './toDeckLayer';
import { loadUserExampleFiles, type UserExample } from './userExamples';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

const REFRESH = 'https://backend.i-guide.io/api/refresh-token';
setRefreshUrl(REFRESH);
(globalThis as unknown as { location: unknown }).location = { origin: 'https://agent.i-guide.io' };
const cfg = { endpoint: '/agent/chat/stream', uploadEndpoint: '/agent/files/upload',
              apiKey: '' } as AgentConfig;

/** The fake network. `expired` is whether the access cookie is stale; a successful refresh
 *  renews it, as the platform's refresh endpoint re-mints the cookie. */
let expired = true;
let refreshOk = true;
let refreshDelayMs = 0;
let refreshes = 0;
let requests: { url: string; init?: RequestInit }[] = [];
const FC = { type: 'FeatureCollection', features: [{ type: 'Feature', properties: {},
  geometry: { type: 'Point', coordinates: [-157.8, 21.3] } }] };
const refused = (status: number, reason: string) =>
  new Response(JSON.stringify({ error: 'nope', reason }), { status });

(globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown, init?: RequestInit) => {
  const u = String(url);
  if (u === REFRESH) {
    refreshes++;
    if (refreshDelayMs) await new Promise((r) => setTimeout(r, refreshDelayMs));
    if (refreshOk) expired = false;
    return new Response('{}', { status: refreshOk ? 200 : 401 });
  }
  requests.push({ url: u, init });
  if (init?.signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
  if (u.includes('/forbidden')) return refused(403, 'insufficient_role');
  if (expired) return refused(401, 'token_expired');
  if (u.endsWith('/agent/models')) return new Response(JSON.stringify({ models: [] }), { status: 200 });
  if (u.endsWith('/agent/chat/cancel')) return new Response('{}', { status: 200 });
  return new Response(JSON.stringify(FC), { status: 200 });
};
const reset = (o: { expired?: boolean; refreshOk?: boolean; refreshDelayMs?: number } = {}) => {
  expired = o.expired ?? true;
  refreshOk = o.refreshOk ?? true;
  refreshDelayMs = o.refreshDelayMs ?? 0;
  refreshes = 0;
  requests = [];
};
const LAYER = 'https://agent.i-guide.io/agent/files/file_1/download';
const viaAgent = (u: string) => fetchAgentFile(u, cfg);

// --- the incident: a layer download after the cookie aged out --------------------
reset();
let loaded = await loadLayerData(undefined, LAYER, viaAgent);
eq('an expired layer download refreshes and draws', 'fc' in loaded ? loaded.fc.features.length : loaded, 1);
eq('  ...with exactly one refresh', refreshes, 1);
eq('  ...and exactly one retry', requests.length, 2);
eq('  ...the retry carries the cookie too', requests[1].init?.credentials, 'include');

// --- a fresh cookie costs nothing ------------------------------------------------
reset({ expired: false });
await fetchAgentFile(LAYER, cfg);
eq('a live cookie never refreshes', refreshes, 0);
eq('  ...and downloads once', requests.length, 1);

// --- the bound: a cookie the refresh does not fix must not loop ------------------
reset({ refreshOk: false });
loaded = await loadLayerData(undefined, LAYER, viaAgent);
eq('a failed refresh does not retry the download', requests.length, 1);
eq('  ...and the layer says why it is missing', 'problem' in loaded ? loaded.problem : loaded,
   'HTTP 401, token_expired');

// --- only an expiry refreshes ----------------------------------------------------
reset({ expired: false });
const r403 = await fetchAgentFile('https://agent.i-guide.io/agent/files/forbidden/download', cfg);
eq('a 403 is not refreshed', [r403.status, refreshes, requests.length], [403, 0, 1]);

// --- someone else's host is not ours to refresh for ------------------------------
reset();
const other = await fetchAgentFile('https://tiles.example.org/layer.geojson', cfg);
eq("another host's 401 is returned untouched", [other.status, refreshes, requests.length], [401, 0, 1]);

// --- a burst of expired downloads makes ONE refresh ------------------------------
// The incident's two layers expired together; each refreshing on its own is a thundering herd
// on the platform, and the loser retries against a cookie replaced under it.
reset({ refreshDelayMs: 20 });
const both = await Promise.all([fetchAgentFile(LAYER, cfg),
  fetchAgentFile('https://agent.i-guide.io/agent/files/file_2/download', cfg)]);
eq('two expired downloads share one refresh', refreshes, 1);
eq('  ...and both succeed', both.map((r) => r.status), [200, 200]);

// --- Stop still stops the retry --------------------------------------------------
reset();
const ctl = new AbortController();
ctl.abort();
const aborted = await loadLayerData(undefined, LAYER,
  (u) => fetchAgentFile(u, cfg, { signal: ctl.signal }));
eq('an aborted download stays aborted', aborted, { aborted: true });

// --- the raster path: deck.gl's own loader had no retry --------------------------
(globalThis as unknown as { createImageBitmap: unknown }).createImageBitmap =
  async (b: Blob) => ({ bytes: b.size });
reset();
const img = await loadRasterImage('https://agent.i-guide.io/agent/files/dem/download', viaAgent);
eq('an expired raster image refreshes and loads', [refreshes, requests.length, Boolean(img)], [1, 2, true]);
reset({ refreshOk: false });
let rasterErr = '';
try { await loadRasterImage('https://agent.i-guide.io/agent/files/dem/download', viaAgent); }
catch (e) { rasterErr = String((e as Error).message); }
eq('  ...a refused one is an error deck reports, not an empty image', /HTTP 401/.test(rasterErr), true);
const raster = toDeckLayer({ kind: 'raster', id: 'dem', source: 'analysis', label: 'DEM',
  url: 'https://agent.i-guide.io/agent/files/dem/download', bounds: [-158, 21, -157, 22] } as never,
  async () => ({}) as ImageBitmap) as unknown as { props: { fetch?: unknown } };
eq('  ...and the bitmap layer loads its image through that loader',
   typeof raster.props.fetch, 'function');

// --- the other authenticated calls that had no retry -----------------------------
reset();
eq('Stop after the cookie aged out still reaches the server', await cancelTurn(cfg, 'turn-1'), true);
eq('  ...after one refresh', refreshes, 1);
reset();
eq('the model list refreshes too', await fetchModels(cfg), { models: [] });
reset();
const example = { files: [{ name: 'parcels.geojson', url: '/agent/examples/e1/files/parcels.geojson' }] };
const files = await loadUserExampleFiles(example as unknown as UserExample, cfg);
eq("a shared example's files refresh too", [files.length, refreshes], [1, 1]);

// --- the class, not the instance -------------------------------------------------
// Every authenticated agent call in these modules goes through one of the two retries. The two
// left bare are deliberate: ui-config is public, and whoami answers 200 and refreshes itself.
const BARE_OK = ['/agent/ui-config', '/agent/whoami'];
const bare: string[] = [];
for (const file of ['src/agentClient.ts', 'src/userExamples.ts']) {
  const lines = readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, i) => {
    if (!/\bawait fetch\(/.test(line)) return;
    const near = lines.slice(Math.max(0, i - 2), i + 1).join('\n');
    if (/withTokenRetry|fetchWithRefresh/.test(near)) return;
    if (BARE_OK.some((p) => line.includes(p))) return;
    bare.push(`${file}:${i + 1}`);
  });
}
eq('no authenticated agent call is left without a refresh', bare, []);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
