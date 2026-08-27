// In-browser spatial analysis -- stands in for the agent's sandboxed geoprocessing.
// In production these ops run server-side (GeoPandas/Shapely) and the agent
// returns the result as a LayerArtifact; the browser just renders it.
import * as turf from '@turf/turf';
import type { FeatureCollection, Feature, Geometry } from 'geojson';

export interface AnalysisResult {
  fc: FeatureCollection;
  summary: string;
}

// Buffer every feature in a collection by `km` kilometers.
export function bufferFC(fc: FeatureCollection, km: number): AnalysisResult {
  const out: Feature[] = [];
  for (const f of fc.features) {
    const b = turf.buffer(f as any, km, { units: 'kilometers' });
    if (b) out.push(b as Feature);
  }
  return {
    fc: { type: 'FeatureCollection', features: out },
    summary: `Buffered ${fc.features.length} feature(s) by ${km} km.`,
  };
}

// Keep only features that fall inside a region geometry (clip-by-selection).
export function clipToRegion(fc: FeatureCollection, region: Geometry): AnalysisResult {
  const reg = turf.feature(region);
  const kept = fc.features.filter((f) => {
    try {
      return turf.booleanIntersects(f as any, reg);
    } catch {
      return false;
    }
  });
  return {
    fc: { type: 'FeatureCollection', features: kept },
    summary: `Clipped to region: ${kept.length} of ${fc.features.length} feature(s) inside.`,
  };
}

// Convex hull of points (e.g., extent of Overpass results or KB centroids).
export function convexHull(fc: FeatureCollection): AnalysisResult {
  const hull = turf.convex(fc);
  return {
    fc: { type: 'FeatureCollection', features: hull ? [hull] : [] },
    summary: hull ? 'Computed convex hull of features.' : 'Not enough points for a hull.',
  };
}

// Simple point stats over a collection.
export function stats(fc: FeatureCollection, regionKm2?: number): string {
  const n = fc.features.length;
  const pts = fc.features.filter((f) => f.geometry?.type === 'Point').length;
  const parts = [`${n} feature(s)` + (pts ? `, ${pts} point(s)` : '')];
  if (regionKm2 != null) {
    parts.push(`region ≈ ${regionKm2.toFixed(0)} km²`);
    if (pts > 0 && regionKm2 > 0) parts.push(`density ≈ ${(pts / regionKm2).toFixed(3)} /km²`);
  }
  return parts.join(' · ');
}

export function areaKm2(region: Geometry): number {
  try {
    return turf.area(turf.feature(region)) / 1e6;
  } catch {
    return 0;
  }
}

// --- Spatial relationship selection (the "rivers that interact with X" case) ---
function bboxOverlap(a: number[], b: number[]): boolean {
  return !(a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]);
}

export function layerBBox(fc: FeatureCollection): [number, number, number, number] {
  return turf.bbox(fc) as [number, number, number, number];
}

// Keep candidate features that spatially relate to ANY feature in target.
// bbox pre-filter keeps this cheap even for large target layers (e.g. 800 tracts).
export function selectRelated(
  candidates: FeatureCollection,
  target: FeatureCollection,
  relation: 'intersects' | 'within',
): AnalysisResult {
  const tFeats = target.features.map((f) => ({ f, bb: safeBBox(f) })).filter((t) => t.bb);
  const out: Feature[] = [];
  for (const c of candidates.features) {
    const cb = safeBBox(c);
    if (!cb) continue;
    for (const t of tFeats) {
      if (!bboxOverlap(cb, t.bb!)) continue;
      try {
        const hit = relation === 'within'
          ? turf.booleanWithin(c as any, t.f as any)
          : turf.booleanIntersects(c as any, t.f as any);
        if (hit) { out.push(c); break; }
      } catch { /* skip degenerate geometry */ }
    }
  }
  return {
    fc: { type: 'FeatureCollection', features: out },
    summary: `${out.length} of ${candidates.features.length} feature(s) ${relation === 'within' ? 'within' : 'intersecting'} the target layer.`,
  };
}

function safeBBox(f: Feature): number[] | null {
  try { return turf.bbox(f); } catch { return null; }
}
