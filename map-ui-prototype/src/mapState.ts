/** What is on the map, sent with every chat turn as `map_state` (Stage 81).
 *
 * The agent cannot see this map. It used to guess from its own record of the layers it had
 * SENT, and that guess was wrong live on 2026-10-18 both ways: two layers that failed to load
 * (401 token_expired) were still "available on your interactive map" in the next answer, and a
 * buffer the agent had computed but never drawn counted as drawn, so "show the site, its buffer
 * and the schools" ended with nothing new on the map. The browser knows; this says it.
 *
 * Kept small on purpose: no geometry, and both lists capped, with the totals sent when capped.
 * Pure, so `npm run check:mapstate` can run it. */
import type { LayerArtifact } from './contracts';

export const MAP_STATE_MAX_LAYERS = 40;
export const MAP_STATE_MAX_UNDRAWN = 20;
const LABEL_CHARS = 120;

type BBox = [number, number, number, number];

export interface MapStateLayer {
  id: string;
  label: string;
  kind: 'geojson' | 'raster';
  render?: string;
  features?: number;
  visible: boolean;
  opacity?: number;
  source?: string;
}

/** A layer the agent sent that did not draw, and why (layerLoad.ts). */
export interface UndrawnLayer { id?: string; label: string; reason: string }

export interface MapState {
  layers: MapStateLayer[];
  undrawn: UndrawnLayer[];
  viewport?: BBox;
  region?: BBox;
  /** Present only when a list was cut: the totals before the cut. */
  truncated?: { layers: number; undrawn: number };
}

const cut = (s: string) => (s.length <= LABEL_CHARS ? s : `${s.slice(0, LABEL_CHARS - 1)}…`);
const round4 = (b: BBox): BBox => b.map((n) => Math.round(n * 1e4) / 1e4) as BBox;
const finiteBox = (b?: BBox | null): b is BBox =>
  Array.isArray(b) && b.length === 4 && b.every((n) => Number.isFinite(n));

function describe(l: LayerArtifact): MapStateLayer {
  const out: MapStateLayer = { id: l.id, label: cut(l.label || l.id), kind: l.kind,
                               visible: l.visible !== false };
  if (l.source) out.source = l.source;
  if (l.kind === 'raster') {
    out.render = 'raster';
    if (l.opacity != null) out.opacity = l.opacity;
  } else {
    out.render = l.render || 'geojson';
    // A sampled layer is SHOWING `shown` features; the total is what the data held.
    out.features = l.partial ? l.partial.shown : (l.data?.features?.length ?? 0);
    const opacity = l.fillOpacity ?? l.style?.opacity;
    if (opacity != null) out.opacity = opacity;
  }
  return out;
}

export function buildMapState(
  layers: LayerArtifact[],
  undrawn: UndrawnLayer[],
  viewport?: BBox | null,
  region?: BBox | null,
): MapState {
  // A layer that drew after an earlier failure (a retry, a restored session) is on the map.
  const held = new Set(layers.map((l) => l.id));
  const failed = undrawn.filter((u) => !u.id || !held.has(u.id))
    .map((u) => ({ ...(u.id ? { id: u.id } : {}), label: cut(u.label), reason: cut(u.reason) }));
  const state: MapState = {
    layers: layers.slice(0, MAP_STATE_MAX_LAYERS).map(describe),
    undrawn: failed.slice(0, MAP_STATE_MAX_UNDRAWN),
  };
  if (layers.length > MAP_STATE_MAX_LAYERS || failed.length > MAP_STATE_MAX_UNDRAWN) {
    state.truncated = { layers: layers.length, undrawn: failed.length };
  }
  if (finiteBox(viewport)) state.viewport = round4(viewport);
  if (finiteBox(region)) state.region = round4(region);
  return state;
}
