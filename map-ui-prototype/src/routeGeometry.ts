// Where a route starts and ends, from its line features alone (no deck.gl, so it can be checked
// in node: routeGeometry.check.ts).
import type { FeatureCollection, Position } from 'geojson';

export function lineParts(fc: FeatureCollection): Position[][] {
  const out: Position[][] = [];
  for (const f of fc.features) {
    const g: any = f.geometry;
    if (g?.type === 'LineString') out.push(g.coordinates);
    else if (g?.type === 'MultiLineString') out.push(...g.coordinates);
  }
  return out.filter((p) => p.length >= 2);
}

/** The two ends of a path: the vertices that end exactly one of its segments. A route written
 *  as one edge per feature need not come in travel order, so the first and last coordinates of
 *  the file are not enough; the ends of the path are the only odd-degree vertices. The start is
 *  the end the file begins at, when it begins at one. Null when the lines do not form one path. */
export function routeEnds(fc: FeatureCollection): { start: Position; end: Position } | null {
  const parts = lineParts(fc);
  if (!parts.length) return null;
  const key = (c: Position) => `${(+c[0]).toFixed(7)},${(+c[1]).toFixed(7)}`;
  const seen = new Map<string, { c: Position; n: number }>();
  for (const p of parts) {
    for (const c of [p[0], p[p.length - 1]]) {
      const k = key(c);
      const e = seen.get(k);
      if (e) e.n += 1; else seen.set(k, { c, n: 1 });
    }
  }
  const odd = [...seen.values()].filter((e) => e.n % 2 === 1).map((e) => e.c);
  if (odd.length !== 2) return null;
  const first = key(parts[0][0]);
  const [s, e] = key(odd[1]) === first ? [odd[1], odd[0]] : [odd[0], odd[1]];
  return { start: s, end: e };
}
