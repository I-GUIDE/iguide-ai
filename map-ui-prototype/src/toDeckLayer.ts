// The thin renderer: LayerArtifact -> deck.gl layer. One switch, easy to test.
import { BitmapLayer, GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import type { LayerArtifact, RGBA } from './contracts';
import type { Feature, Point } from 'geojson';
import { routeEnds } from './routeGeometry';
import { fillAndLine, resolveLegend } from './palette';

const DEFAULT_FILL: [number, number, number, number] = [51, 136, 255, 120];
const DEFAULT_LINE: [number, number, number, number] = [20, 90, 200, 255];

function centroidOf(f: Feature): [number, number] | null {
  const g = f.geometry;
  if (!g) return null;
  if (g.type === 'Point') return (g as Point).coordinates as [number, number];
  return null;
}

// Sequential ramp for a choropleth: light -> deep, computed over the layer's own range.
type VectorLayer = Extract<LayerArtifact, { kind: 'geojson' }>;

function rampFor(a: VectorLayer): ((f: any) => [number, number, number, number]) | null {
  if (!a.styleBy) return null;
  const key = a.styleBy;
  const vals = a.data.features
    .map((f) => Number((f.properties || {})[key]))
    .filter((v) => Number.isFinite(v));
  if (!vals.length) return null;
  const min = Math.min(...vals), max = Math.max(...vals);
  const span = max - min || 1;
  return (f: any) => {
    const v = Number((f.properties || {})[key]);
    const t = Number.isFinite(v) ? Math.min(1, Math.max(0, (v - min) / span)) : 0;
    // #fff5eb -> #7f2704 (OrRd), the conventional choropleth ramp
    return [255 - Math.round(128 * t), 245 - Math.round(206 * t), 235 - Math.round(231 * t), 200];
  };
}

// Categorical fill: look the feature's class up in the legend. A class keeps the colour the
// tool sent when it sent one (it means something, or the user asked); the rest take the map's
// palette (palette.ts), so a new categorical statistic needs no change in this file.
function categoricalFor(a: VectorLayer): ((f: any) => [number, number, number, number]) | null {
  if (!a.styleBy || !a.legend?.length) return null;
  const key = a.styleBy;
  const byLabel = new Map(resolveLegend(a.legend).map((e) => [e.label, e.color]));
  const UNKNOWN: [number, number, number, number] = [200, 200, 200, 140];
  return (f: any) => byLabel.get(String((f.properties || {})[key])) ?? UNKNOWN;
}

// Navigation-app styling for a road network and a route found on it. The network is context:
// thin slate lines the basemap still reads through. The route is the answer: a thick blue line
// on a white casing, so it stays legible over any street, with its start (green) and end (red)
// marked. Before this both were drawn with the same 2-3 px default line, and the fastest drive
// across the Loop could not be told apart from the 1,521 ways it was found on.
const NETWORK_LINE: [number, number, number, number] = [71, 85, 105, 150];
const ROUTE_LINE: [number, number, number, number] = [26, 115, 232, 255];
const ROUTE_CASING: [number, number, number, number] = [255, 255, 255, 235];
const ROUTE_START: [number, number, number, number] = [52, 168, 83, 255];
const ROUTE_END: [number, number, number, number] = [234, 67, 53, 255];

function routeLayers(a: VectorLayer) {
  const visible = a.visible !== false;
  const line = { lineWidthUnits: 'pixels' as const, lineCapRounded: true, lineJointRounded: true,
                 filled: false, stroked: true };
  const main: [number, number, number, number] = a.color ? [a.color[0], a.color[1], a.color[2], 255] : ROUTE_LINE;
  const layers: any[] = [
    new GeoJsonLayer({ id: `${a.id}--casing`, data: a.data, visible, pickable: false, ...line,
                       getLineColor: ROUTE_CASING, getLineWidth: 11 }),
    new GeoJsonLayer({ id: a.id, data: a.data, visible, pickable: true, ...line,
                       getLineColor: main, getLineWidth: 6,
                       updateTriggers: { getLineColor: [a.color] } }),
  ];
  const ends = routeEnds(a.data);
  if (ends) {
    layers.push(new ScatterplotLayer({
      id: `${a.id}--ends`, visible, pickable: false,
      data: [{ position: ends.start, color: ROUTE_START }, { position: ends.end, color: ROUTE_END }],
      getPosition: (d: any) => d.position, getFillColor: (d: any) => d.color,
      getLineColor: [255, 255, 255, 255], stroked: true, lineWidthUnits: 'pixels', getLineWidth: 3,
      radiusUnits: 'pixels', getRadius: 8,
    }));
  }
  return layers;
}

function networkLayer(a: VectorLayer) {
  return new GeoJsonLayer({
    id: a.id, data: a.data, visible: a.visible !== false, pickable: true,
    filled: false, stroked: true, lineWidthUnits: 'pixels', lineCapRounded: true,
    lineJointRounded: true, getLineColor: a.color ?? NETWORK_LINE, getLineWidth: 1.5,
    updateTriggers: { getLineColor: [a.color] },
    pointType: 'circle', getPointRadius: 2, pointRadiusUnits: 'pixels',
  });
}

// Dots shrink as a layer gets denser. Live 2026-10-17, 10,008 crash points drawn at the default
// 6 px fused into one solid red mass over the city; at 2 px the same points show where they
// cluster. A style that asks for smaller dots keeps them.
export function densityRadius(n: number): number {
  return n > 5000 ? 2 : n > 1500 ? 3 : n > 400 ? 4.5 : 6;
}

/** How opaque a vector layer's fill is drawn now (0..1): the agent's or the user's choice, or
 *  the default the renderer would use. The layer panel's slider starts from it. */
export function fillOpacityOf(a: LayerArtifact): number {
  if (a.kind !== 'geojson') return a.opacity ?? 0.85;
  if (typeof a.fillOpacity === 'number') return a.fillOpacity;
  if (a.styleBy && a.render !== 'categories') return 200 / 255;   // the choropleth ramp's alpha
  if (a.color) return Math.min(a.color[3], 190) / 255;            // fillAndLine's cap
  return (a.style?.fill?.[3] ?? 120) / 255;
}

/** Whether a layer has a fill to make see-through: lines and density surfaces do not. */
export function hasFill(a: LayerArtifact): boolean {
  return a.kind === 'geojson' && !a.outline
    && !['route', 'network', 'heatmap'].includes(a.render ?? '');
}

/** How a raster's image is fetched: the agent's own files go through fetchAgentFile, which
 *  carries the page's credentials and refreshes an expired sign-in once. */
export type FetchFile = (url: string, init?: RequestInit) => Promise<Response>;

/** A raster's image, fetched through `fetchFile` rather than by deck.gl's loader. deck loads a
 *  string `image` itself, outside every retry this page has, so an image requested after the
 *  one-hour access cookie aged out was refused and the raster never drew. */
export async function loadRasterImage(url: string, fetchFile: FetchFile,
                                      signal?: AbortSignal): Promise<ImageBitmap> {
  const res = await fetchFile(url, signal ? { signal } : {});
  if (!res.ok) throw new Error(`raster image ${url}: HTTP ${res.status}`);
  return createImageBitmap(await res.blob());
}

/** Loads a raster's image by url; see loadRasterImage. */
export type LoadImage = (url: string, signal?: AbortSignal) => Promise<ImageBitmap>;

export function toDeckLayer(a: LayerArtifact, loadImage?: LoadImage) {
  // A raster is an image with a footprint, not geometry: deck.gl wants the bounds in
  // [left, bottom, right, top] order, which is the same order the agent sends.
  if (a.kind === 'raster') {
    return new BitmapLayer({
      id: a.id,
      image: a.url,
      bounds: a.bounds,
      opacity: a.opacity ?? 0.85,
      visible: a.visible !== false,
      pickable: false,
      // deck calls `fetch` for a string `image` in place of its own loader.
      ...(loadImage ? {
        fetch: (url: string, ctx: { signal?: AbortSignal }) => loadImage(url, ctx?.signal),
      } : {}),
    });
  }

  const s = a.style ?? {};
  // A colour the agent set because the user asked for it wins over the source's default.
  const asked = a.kind === 'geojson' && a.color ? fillAndLine(a.color) : null;
  const fill = asked?.fill ?? s.fill ?? DEFAULT_FILL;
  const line = asked?.line ?? s.line ?? DEFAULT_LINE;

  if (a.render === 'heatmap') {
    const pts = a.data.features
      .map((f) => centroidOf(f))
      .filter((c): c is [number, number] => !!c)
      .map((c) => ({ position: c, weight: 1 }));
    return new HeatmapLayer({
      id: a.id,
      visible: a.visible !== false,
      data: pts,
      getPosition: (d: any) => d.position,
      getWeight: (d: any) => d.weight,
      // Tuned against 31,977 Chicago incidents, A/B'd in the browser: 40px kernels with no
      // threshold settle into a soft yellow mass that shows the city's outline but almost no
      // internal structure, because every pixel carries some weight. A tighter kernel resolves
      // neighbourhood-level hotspots, and the threshold drops the cold tail entirely so the
      // basemap reads through where there is nothing to report.
      radiusPixels: 22,
      // One hue, light to full, when the user asked for a colour; deck's default ramp otherwise.
      ...(a.color ? { colorRange: [0.15, 0.35, 0.55, 0.7, 0.85, 1].map((t) =>
        [a.color![0], a.color![1], a.color![2], Math.round(255 * t)]) } : {}),
      intensity: 1,
      threshold: 0.08,
      opacity: s.opacity ?? 0.8,
      pickable: false,
    });
  }

  if (a.render === 'route') return routeLayers(a);
  if (a.render === 'network') return networkLayer(a);

  // Categorical first: a class-name column would ramp to NaN and come out one flat colour.
  const ramp = a.render === 'categories' ? categoricalFor(a) ?? rampFor(a) : rampFor(a);
  // The agent's fill opacity, when it set one, replaces the fill's alpha (the outline keeps
  // its own), so a framing layer shows the map and the results through it.
  const alpha = typeof a.fillOpacity === 'number' ? Math.round(255 * a.fillOpacity) : null;
  const withAlpha = (c: RGBA): RGBA => (alpha === null ? c : [c[0], c[1], c[2], alpha]);
  const fillColor = alpha === null ? (ramp ?? fill)
    : ramp ? (f: any) => withAlpha(ramp(f)) : withAlpha(fill);
  return new GeoJsonLayer({
    id: a.id,
    data: a.data,
    visible: a.visible !== false,
    pickable: true,
    stroked: true,
    // An outline layer frames what is underneath instead of covering it — the case that
    // matters is a zone boundary over its own pixel-embedding raster.
    filled: !a.outline,
    extruded: !!s.extruded,
    getElevation: s.elevation ?? 0,
    pointType: 'circle',
    getFillColor: fillColor,
    // On a choropleth the FILL carries the number, so the outline must stay out of its way:
    // a 708-cell hex grid drawn with the standard 2px purple border read as an empty mesh —
    // the borders covered more pixels than the shaded interiors. Hairline grey instead.
    getLineColor: a.outline ? (s.line ?? [90, 60, 160, 255]) : (ramp ? [90, 90, 105, 90] : line),
    getLineWidth: a.outline ? (s.lineWidth ?? 2.5)
                            : (ramp ? (s.lineWidth ?? 0.5) : (s.lineWidth ?? 2)),
    lineWidthUnits: 'pixels',
    getPointRadius: Math.min(s.pointRadius ?? 6, densityRadius(a.data.features.length)),
    pointRadiusUnits: s.radiusUnits ?? 'pixels',
    opacity: s.opacity ?? 1,
    // expose source + props for the tooltip
    updateTriggers: { getFillColor: [a.styleBy, a.data, a.render, a.legend, a.color, a.fillOpacity],
                      getLineColor: [a.color] },
  });
}
