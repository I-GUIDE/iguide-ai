// The agent's point size, highlight, labels and line options reach the deck layers as sent, and
// a layer without them draws exactly as before (Stage 80).
import { highlightTest, sizeScale, stylingFromWire } from './layerStyling';
import { densityRadius, toDeckLayer } from './toDeckLayer';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };

const pt = (lon: number, lat: number, props: Record<string, any>) => ({
  type: 'Feature' as const, properties: props, geometry: { type: 'Point' as const, coordinates: [lon, lat] } });
const quakes = { type: 'FeatureCollection' as const, features: [
  pt(-117.5, 35.7, { mag: 3.0, kind: 'aftershock' }), pt(-117.6, 35.77, { mag: 7.1, kind: 'mainshock' }),
  pt(-117.55, 35.72, { mag: 5.05, kind: 'aftershock' }), pt(-117.52, 35.71, { mag: null, kind: 'aftershock' }) ] };
const base = { kind: 'geojson' as const, id: 'q', source: 'analysis', label: 'quakes', data: quakes,
               style: { pointRadius: 6 } };

// --- wire parsing -------------------------------------------------------------------------
const wire = stylingFromWire({
  size_by: { column: 'mag', min: 3, max: 7.1, range: [3, 16] },
  highlight: { column: 'mag', op: 'gte', value: 7, numeric: true, label: 'mainshock', color: [255, 193, 7, 255], count: 1 },
  labels: { column: 'mag', scope: 'highlighted', items: [{ position: [-117.6, 35.77], text: '7.1' }] },
  line_width: 3, line_color: [0, 0, 0, 255],
});
ok(wire.sizeBy?.range[1] === 16 && wire.highlight?.count === 1 && wire.labels?.items.length === 1
   && wire.lineWidth === 3 && wire.lineColor?.[3] === 255, 'well-formed fields arrive');
ok(Object.keys(stylingFromWire({})).length === 0, 'nothing set, nothing added');
const bad = stylingFromWire({ size_by: { column: 'mag', min: 'x', max: 2, range: [3, 16] },
  highlight: { column: 'mag', op: 'like', value: 7 }, labels: { items: [{ position: [1], text: 'a' }] },
  point_size: -2, line_width: 'wide' });
ok(Object.keys(bad).length === 0, 'malformed fields are dropped, not guessed at');

// --- the predicate and the scale ----------------------------------------------------------
const gte7 = highlightTest({ column: 'mag', op: 'gte', value: 7, numeric: true, label: '', color: [0, 0, 0, 255] });
ok(quakes.features.filter((f) => gte7(f.properties)).length === 1, 'numeric >= picks the mainshock only');
const isMain = highlightTest({ column: 'kind', op: 'eq', value: 'mainshock', numeric: false, label: '', color: [0, 0, 0, 255] });
ok(quakes.features.filter((f) => isMain(f.properties)).length === 1, 'text equals picks one');
const either = highlightTest({ column: 'mag', op: 'in', value: [3, 7.1], numeric: true, label: '', color: [0, 0, 0, 255] });
ok(quakes.features.filter((f) => either(f.properties)).length === 2, 'in picks each listed value');
ok(!gte7({ mag: null }) && !gte7({}), 'a missing value never matches');
const at = sizeScale({ column: 'mag', min: 3, max: 7.1, range: [3, 16] });
ok(at({ mag: 3 }) === 3 && at({ mag: 7.1 }) === 16, 'the range ends map to the pixel ends');
ok(Math.abs(at({ mag: 5.05 }) - (3 + 13 * Math.sqrt(0.5))) < 1e-9, 'square-root scale in between');
ok(at({ mag: null }) === 3 && at({ mag: 99 }) === 16, 'missing -> smallest; out of range clamps');

// --- the deck layers ----------------------------------------------------------------------
const plain = toDeckLayer(base) as any;
ok(!Array.isArray(plain) && plain.props.getPointRadius === Math.min(6, densityRadius(4)),
   'no option: one layer, the same constant radius as before');
ok(toDeckLayer({ ...base, pointSize: 9 } as any) && (toDeckLayer({ ...base, pointSize: 9 } as any) as any).props.getPointRadius === 9,
   'point_size: one fixed radius');
const sized = (toDeckLayer({ ...base, sizeBy: wire.sizeBy } as any) as any).props.getPointRadius;
ok(typeof sized === 'function' && sized(quakes.features[1]) === 16 && sized(quakes.features[0]) === 3,
   'size_by: radius per feature');
const all = toDeckLayer({ ...base, sizeBy: wire.sizeBy, highlight: wire.highlight, labels: wire.labels } as any) as any[];
ok(Array.isArray(all) && all.map((l) => l.id).join() === 'q,q--highlight,q--labels',
   'highlight and labels draw over their layer, in that order');
const hl = all[1];
ok(hl.props.data.features.length === 1 && hl.props.data.features[0].properties.mag === 7.1, 'the highlight layer holds the match');
ok(hl.props.getPointRadius(quakes.features[1]) > sized(quakes.features[1]), 'the highlighted point is drawn larger');
ok(all[2].props.data[0].text === '7.1', 'labels carry the server text');
const off = all[2].props.getPixelOffset(all[2].props.data[0]);
ok(off[0] > hl.props.getPointRadius(quakes.features[1]), 'a point label starts past its (highlighted) circle');
const loose = toDeckLayer({ ...base, labels: { column: 'x', scope: 'all', items: [{ position: [0, 0], text: 'Zone' }] } } as any) as any[];
ok(loose[1].props.getPixelOffset(loose[1].props.data[0])[0] === 0
   && loose[1].props.getTextAnchor(loose[1].props.data[0]) === 'middle', 'a label on no point is centred');
const lined = toDeckLayer({ ...base, lineWidth: 3, lineColor: [0, 0, 0, 255] } as any) as any;
ok(lined.props.getLineWidth === 3 && lined.props.getLineColor[0] === 0, 'line width and colour apply');
const route = toDeckLayer({ ...base, render: 'route', lineWidth: 4, lineColor: [9, 9, 9, 255] } as any) as any[];
ok(route[1].props.getLineWidth === 4 && route[0].props.getLineWidth === 9 && route[1].props.getLineColor[0] === 9,
   'a route takes the line width (casing 5 px wider) and colour');

if (failed) { console.error(`${failed} styling check(s) failed`); process.exit(1); }
console.log('layer styling ok');
