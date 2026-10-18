// The ends of a route, whatever order its edges were written in.
import type { FeatureCollection } from 'geojson';
import { routeEnds } from './routeGeometry';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };
const line = (...c: number[][]) => ({ type: 'Feature' as const, properties: {}, geometry: { type: 'LineString' as const, coordinates: c } });
const fc = (...f: any[]): FeatureCollection => ({ type: 'FeatureCollection', features: f });

// One merged line: its own first and last vertices.
let e = routeEnds(fc(line([0, 0], [1, 0], [1, 1])));
ok(e && e.start[0] === 0 && e.end[1] === 1, 'one line: first and last vertex');

// One edge per feature, out of travel order: the ends are the odd-degree vertices, and the start
// is the end the file begins at.
e = routeEnds(fc(line([0, 0], [1, 0]), line([2, 0], [3, 0]), line([1, 0], [2, 0])));
ok(e && e.start[0] === 0 && e.end[0] === 3, `edges out of order: start 0, end 3 (got ${JSON.stringify(e)})`);

// An edge written backwards still joins the path.
e = routeEnds(fc(line([1, 0], [0, 0]), line([1, 0], [2, 0])));
ok(e && [e.start[0], e.end[0]].sort().join() === '0,2', 'a reversed edge still joins');

// Not one path (two separate pieces): no markers rather than wrong ones.
ok(routeEnds(fc(line([0, 0], [1, 0]), line([5, 5], [6, 6]))) === null, 'two pieces: no ends');
ok(routeEnds(fc()) === null, 'no lines: no ends');

if (failed) { console.error(`${failed} route check(s) failed`); process.exit(1); }
console.log('route ends ok');
