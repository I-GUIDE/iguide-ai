/** A runnable check that a map layer which fails to load says so. `npm run check:layers`.
 *
 * Same reasoning as check:files: no test runner here, and this failure leaves nothing a
 * screenshot can explain. On 2026-10-17 (round r59) the route GeoJSON held bare NaN, JSON.parse
 * threw "Unexpected token 'N'", an empty `catch {}` in onMapLayer swallowed it, and the answer
 * said the route was on a map that had nothing on it. The body below is the r59 shape. */
import { loadLayerData, undrawnLine, type LayerLoad } from './layerLoad';

let bad = 0;
const ok = (label: string, cond: boolean, got: unknown) => {
  if (!cond) bad++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};
const problem = (r: LayerLoad) => ('problem' in r ? r.problem : '');

const route = (streets: string) => `{"type":"FeatureCollection","features":[{"type":"Feature",`
  + `"properties":{"time_min":5.19,"streets":${streets}},`
  + `"geometry":{"type":"LineString","coordinates":[[-87.645,41.866],[-87.618,41.889]]}}]}`;
const serve = (body: string, status = 200) => async () =>
  new Response(body, { status, headers: { 'Content-Type': 'application/json' } });

// The r59 file, as Python's json wrote it.
const nan = await loadLayerData(undefined, '/agent/files/f/download',
                                serve(route('["W Van Buren St", NaN, "E Wacker Dr"]')));
ok('a NaN file is a problem, not a silence', 'problem' in nan, nan);
ok('  ...that names the parse failure', /not valid JSON/.test(problem(nan)), problem(nan));
const line = undrawnLine('fastest route', problem(nan));
ok('  ...and the line says it is not on the map',
   line.startsWith('map: fastest route could not be loaded (') && line.endsWith('so it is not on the map'),
   line);

// The same file once the server has made it strict draws.
const strict = await loadLayerData(undefined, '/agent/files/f/download',
                                   serve(route('["W Van Buren St", null, "E Wacker Dr"]')));
ok('the strict file draws', 'fc' in strict && strict.fc.features.length === 1, Object.keys(strict));

// The refusal the HTTP branch already reported keeps its wording.
const refused = await loadLayerData(undefined, '/agent/files/f/download',
                                    serve('{"error":"x","reason":"not_signed_in"}', 403));
ok('a refused download keeps its status and reason', problem(refused) === 'HTTP 403, not_signed_in',
   problem(refused));

const empty = await loadLayerData(undefined, '/u', serve('{"type":"FeatureCollection","features":[]}'));
ok('an empty file is said, not skipped', problem(empty) === 'its file holds no features', problem(empty));

const notFc = await loadLayerData(undefined, '/u', serve('{"type":"Point","coordinates":[0,0]}'));
ok('a non-collection is said', /not a GeoJSON FeatureCollection/.test(problem(notFc)), problem(notFc));

const netFail = await loadLayerData(undefined, '/u', async () => { throw new TypeError('Failed to fetch'); });
ok('a network failure is said', problem(netFail) === 'the download failed: Failed to fetch', problem(netFail));

// Stop is the reader's own act: nothing to warn about.
const stopped = await loadLayerData(undefined, '/u', async () => {
  throw Object.assign(new Error('aborted'), { name: 'AbortError' });
});
ok('Stop is not reported as a failure', 'aborted' in stopped, stopped);

// An inline event: drawn without a fetch, and an event with nothing in it is said.
const inline = await loadLayerData(JSON.parse(route('[]')), undefined,
                                   async () => { throw new Error('must not fetch'); });
ok('an inline layer draws without a fetch', 'fc' in inline, Object.keys(inline));
const nothing = await loadLayerData(undefined, undefined, async () => { throw new Error('no'); });
ok('an event with no data and no url is said', /no data/.test(problem(nothing)), problem(nothing));

if (bad) { console.error(`\n${bad} check(s) failed`); process.exit(1); }
console.log('\nall layer-load checks passed');
