// Classes without a colour get the map's palette; colours the agent sent always win.
import { CATEGORY_PALETTE, resolveLegend } from './palette';
import { toDeckLayer } from './toDeckLayer';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };

const plain = resolveLegend([{ label: 'A' }, { label: 'B' }, { label: 'C' }]);
ok(plain.map((e) => e.color.join()).join('|') === CATEGORY_PALETTE.slice(0, 3).map((c) => c.join()).join('|'),
   'classes without colours take the palette in order');

const red: [number, number, number, number] = [255, 0, 0, 255];
const mixed = resolveLegend([{ label: 'A' }, { label: 'Lozano', color: red }, { label: 'C' }]);
ok(mixed[1].color === red, 'a colour the agent sent is kept');
ok(mixed[0].color.join() === CATEGORY_PALETTE[0].join(), 'the first free hue goes to the first plain class');

// The palette's first hue is a red; asking for exactly that red for one class must not leave a
// second class in the same red.
const first = CATEGORY_PALETTE[0];
const clash = resolveLegend([{ label: 'A', color: first }, { label: 'B' }]);
ok(clash[1].color.join() !== first.join(), 'an explicit colour is not reused for another class');

ok(resolveLegend(undefined).length === 0 && resolveLegend([]).length === 0, 'no legend, no entries');

// A fill opacity the agent chose replaces the fill's alpha; the outline keeps its own.
const square = { type: 'FeatureCollection' as const, features: [{ type: 'Feature' as const, properties: { v: 1 },
  geometry: { type: 'Polygon' as const, coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]] } }] };
const base = { kind: 'geojson' as const, id: 'buf', source: 'analysis', label: 'buffer', data: square };
const fillOf = (l: any) => (toDeckLayer(l) as any).props.getFillColor;
const lineOf = (l: any) => (toDeckLayer(l) as any).props.getLineColor;
ok(fillOf({ ...base, color: red, fillOpacity: 0.15 })[3] === 38, 'fillOpacity 0.15 -> alpha 38');
ok(lineOf({ ...base, color: red, fillOpacity: 0.15 })[3] === 255, 'the outline keeps full alpha');
ok(fillOf({ ...base, color: red })[3] === 190, 'no fillOpacity: the colour cap stays');
const ramped = fillOf({ ...base, styleBy: 'v', fillOpacity: 0.5 });
ok(typeof ramped === 'function' && ramped(square.features[0])[3] === 128, 'a ramp takes the chosen alpha too');

if (failed) { console.error(`${failed} palette check(s) failed`); process.exit(1); }
console.log('palette ok');
