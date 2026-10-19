/** A runnable check of the `map_state` each turn sends. `npm run check:mapstate`.
 *
 * The two live failures of 2026-10-18 it exists for: layers that failed to load (401
 * token_expired) still read as on the map, and a schools-only map read as "site, buffer and
 * schools drawn". Both need the report to say exactly what drew and what did not. */
import type { FeatureCollection } from 'geojson';
import type { LayerArtifact } from './contracts';
import { buildMapState, MAP_STATE_MAX_LAYERS } from './mapState';
import { undrawnLine } from './layerLoad';

let bad = 0;
const ok = (label: string, cond: boolean, got: unknown) => {
  if (!cond) bad++;
  console.log(`${cond ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

const pts = (n: number): FeatureCollection => ({ type: 'FeatureCollection', features: Array.from(
  { length: n }, (_, i) => ({ type: 'Feature', properties: { i },
                              geometry: { type: 'Point', coordinates: [-87.62 + i / 1000, 41.88] } })) });
const schools: LayerArtifact = { kind: 'geojson', id: 'schools', source: 'analysis',
                                 label: 'Schools within 1 mile', data: pts(20), fillOpacity: 0.4 };
const hidden: LayerArtifact = { kind: 'geojson', id: 'site', source: 'analysis', label: 'Site',
                                data: pts(1), visible: false, render: 'geojson' };
const raster: LayerArtifact = { kind: 'raster', id: 'pca', source: 'analysis', label: 'clay PCA',
                                url: '/f/1.png', bounds: [-88, 40, -87, 41], opacity: 0.85 };

const s = buildMapState([schools, hidden, raster],
  [{ id: 'suit', label: 'Suitability score', reason: 'HTTP 401, token_expired' }],
  [-87.65432, 41.86, -87.6, 41.9], null);
ok('one entry per layer', s.layers.length === 3, s.layers.map((l) => l.id));
ok('  ...with its feature count, render and source',
   s.layers[0].features === 20 && s.layers[0].render === 'geojson' && s.layers[0].source === 'analysis',
   s.layers[0]);
ok('  ...its fill opacity', s.layers[0].opacity === 0.4, s.layers[0].opacity);
ok('  ...and whether the user hid it', s.layers[1].visible === false && s.layers[0].visible, s.layers[1]);
ok('a raster is reported without features', s.layers[2].render === 'raster'
   && s.layers[2].features === undefined, s.layers[2]);
ok('no geometry is sent', !JSON.stringify(s).includes('coordinates'), JSON.stringify(s).length);
ok('the failed layer is reported with its reason',
   s.undrawn[0]?.label === 'Suitability score' && s.undrawn[0].reason === 'HTTP 401, token_expired',
   s.undrawn);
ok('viewport rounded, no region', s.viewport?.[0] === -87.6543 && s.region === undefined, s);
ok('not capped, so no truncation note', s.truncated === undefined, s.truncated);

// A failure that later drew (same id) is on the map, not failed.
const redrawn = buildMapState([{ ...schools, id: 'suit' }],
  [{ id: 'suit', label: 'Suitability score', reason: 'HTTP 401' }]);
ok('a layer that drew after failing is not reported as failed', redrawn.undrawn.length === 0, redrawn);

// The cap, and saying so.
const many = Array.from({ length: MAP_STATE_MAX_LAYERS + 5 },
                        (_, i) => ({ ...schools, id: `l${i}`, label: `layer ${i}` }));
const capped = buildMapState(many, []);
ok('layers capped', capped.layers.length === MAP_STATE_MAX_LAYERS, capped.layers.length);
ok('  ...and the payload says it was capped', capped.truncated?.layers === MAP_STATE_MAX_LAYERS + 5,
   capped.truncated);
ok('a 45-layer report stays small', JSON.stringify(capped).length < 8000, JSON.stringify(capped).length);

// The reason the trace shows is the one the agent gets.
const reason = 'HTTP 401, token_expired';
ok('undrawnLine and the report carry the same reason',
   undrawnLine('Suitability score', reason).includes(reason), reason);

if (bad) { console.error(`\n${bad} check(s) failed`); process.exit(1); }
console.log('\nall map-state checks passed');
