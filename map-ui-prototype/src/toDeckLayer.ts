// The thin renderer: LayerArtifact -> deck.gl layer. One switch, easy to test.
import { BitmapLayer, GeoJsonLayer, ScatterplotLayer, TextLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import type { LayerArtifact, RGBA } from './contracts';
import type { Feature, Point } from 'geojson';
import { routeEnds } from './routeGeometry';
import { fillAndLine, resolveLegend } from './palette';
import { highlightTest, sizeScale } from './layerStyling';

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
  const main: [number, number, number, number] = a.lineColor ?? (a.color ? [a.color[0], a.color[1], a.color[2], 255] : ROUTE_LINE);
  const width = a.lineWidth ?? 6;
  const layers: any[] = [
    new GeoJsonLayer({ id: `${a.id}--casing`, data: a.data, visible, pickable: false, ...line,
                       getLineColor: ROUTE_CASING, getLineWidth: width + 5,
                       updateTriggers: { getLineWidth: [a.lineWidth] } }),
    new GeoJsonLayer({ id: a.id, data: a.data, visible, pickable: true, ...line,
                       getLineColor: main, getLineWidth: width,
                       updateTriggers: { getLineColor: [a.color, a.lineColor], getLineWidth: [a.lineWidth] } }),
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
    lineJointRounded: true, getLineColor: a.lineColor ?? a.color ?? NETWORK_LINE,
    getLineWidth: a.lineWidth ?? 1.5,
    updateTriggers: { getLineColor: [a.color, a.lineColor], getLineWidth: [a.lineWidth] },
    pointType: 'circle', getPointRadius: 2, pointRadiusUnits: 'pixels',
  });
}

// Dots shrink as a layer gets denser. Live 2026-10-17, 10,008 crash points drawn at the default
// 6 px fused into one solid red mass over the city; at 2 px the same points show where they
// cluster. A style that asks for smaller dots keeps them.
export function densityRadius(n: number): number {
  return n > 5000 ? 2 : n > 1500 ? 3 : n > 400 ? 4.5 : 6;
}

/** A vector layer's point radius in pixels: by the agent's size column, its one fixed size, or
 *  (as always before Stage 80) the style's radius capped by how dense the layer is. */
export function pointRadiusOf(a: VectorLayer): number | ((f: any) => number) {
  if (a.sizeBy) {
    const at = sizeScale(a.sizeBy);
    return (f: any) => at(f?.properties);
  }
  if (typeof a.pointSize === 'number') return a.pointSize;
  return Math.min(a.style?.pointRadius ?? 6, densityRadius(a.data.features.length));
}

const isLine = (f: any) => /LineString/.test(f?.geometry?.type ?? '');
const HIGHLIGHT_EDGE: RGBA = [33, 33, 33, 255];

const highlightRadius = (r: number) => Math.max(r * 1.5, r + 4, 8);

/** The features the agent singled out, drawn again on top of their layer: larger, in the
 *  highlight colour, with a dark edge so they read on any basemap and over any ramp. */
function highlightLayer(a: VectorLayer, baseRadius: number | ((f: any) => number), baseLine: number) {
  const h = a.highlight!;
  const test = highlightTest(h);
  const picked = a.data.features.filter((f) => test(f.properties));
  if (!picked.length) return null;
  const radius = (f: any) => highlightRadius(typeof baseRadius === 'function' ? baseRadius(f) : baseRadius);
  return new GeoJsonLayer({
    id: `${a.id}--highlight`,
    data: { type: 'FeatureCollection', features: picked } as any,
    visible: a.visible !== false,
    pickable: true,
    stroked: true,
    filled: true,
    pointType: 'circle',
    getFillColor: [h.color[0], h.color[1], h.color[2], 230],
    // A highlighted line IS its colour; a point or polygon keeps a dark edge around it.
    getLineColor: (f: any) => (isLine(f) ? h.color : HIGHLIGHT_EDGE),
    getLineWidth: (f: any) => (isLine(f) ? baseLine * 2 + 2 : 2),
    lineWidthUnits: 'pixels',
    lineCapRounded: true,
    lineJointRounded: true,
    getPointRadius: radius,
    pointRadiusUnits: 'pixels',
    updateTriggers: { getPointRadius: [a.sizeBy, a.pointSize], getFillColor: [h.color],
                      getLineColor: [h.color], getLineWidth: [baseLine] },
  });
}

/** The agent's labels, placed by the server (at most 50), on a light backing so they read
 *  over points, polygons and either basemap. A point's label starts just past the circle as
 *  drawn (a highlighted one is larger); any other label is centred on its place. */
function labelLayer(a: VectorLayer, radius: number | ((f: any) => number)) {
  const key = (c: any) => `${Number(c[0]).toFixed(6)},${Number(c[1]).toFixed(6)}`;
  const test = a.highlight ? highlightTest(a.highlight) : null;
  const pointR = new Map<string, number>();
  for (const f of a.data.features) {
    if (f.geometry?.type !== 'Point') continue;
    const r = typeof radius === 'function' ? radius(f) : radius;
    pointR.set(key((f.geometry as Point).coordinates), test?.(f.properties) ? highlightRadius(r) : r);
  }
  const at = (d: any) => pointR.get(key(d.position));
  return new TextLayer({
    id: `${a.id}--labels`,
    data: a.labels!.items,
    visible: a.visible !== false,
    pickable: false,
    getPosition: (d: any) => d.position,
    getText: (d: any) => d.text,
    getSize: 12,
    getColor: [20, 20, 20, 255],
    getTextAnchor: (d: any) => (at(d) === undefined ? 'middle' : 'start'),
    getAlignmentBaseline: 'center',
    getPixelOffset: (d: any) => [at(d) === undefined ? 0 : at(d)! + 5, 0],
    characterSet: 'auto',
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
    background: true,
    getBackgroundColor: [255, 255, 255, 225],
    backgroundPadding: [4, 2],
  });
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

function baseLayers(a: LayerArtifact) {
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
    getLineColor: a.lineColor
      ?? (a.outline ? (s.line ?? [90, 60, 160, 255]) : (ramp ? [90, 90, 105, 90] : line)),
    getLineWidth: lineWidthOf(a, !!ramp),
    lineWidthUnits: 'pixels',
    getPointRadius: pointRadiusOf(a),
    // A size the agent set is in pixels whatever the source's style says.
    pointRadiusUnits: a.sizeBy || typeof a.pointSize === 'number' ? 'pixels' : (s.radiusUnits ?? 'pixels'),
    opacity: s.opacity ?? 1,
    // expose source + props for the tooltip
    updateTriggers: { getFillColor: [a.styleBy, a.data, a.render, a.legend, a.color, a.fillOpacity],
                      getLineColor: [a.color, a.lineColor], getLineWidth: [a.lineWidth],
                      getPointRadius: [a.sizeBy, a.pointSize, a.data] },
  });
}

function lineWidthOf(a: VectorLayer, ramp: boolean): number {
  if (typeof a.lineWidth === 'number') return a.lineWidth;
  const s = a.style ?? {};
  return a.outline ? (s.lineWidth ?? 2.5) : (ramp ? (s.lineWidth ?? 0.5) : (s.lineWidth ?? 2));
}

export function toDeckLayer(a: LayerArtifact) {
  const base = baseLayers(a);
  if (a.kind !== 'geojson' || (!a.highlight && !a.labels?.items.length)) return base;
  // Highlighted features and labels draw over their own layer, in that order.
  const extra: any[] = [];
  if (a.highlight) {
    const line = a.render === 'route' ? (a.lineWidth ?? 6) : lineWidthOf(a, !!a.styleBy);
    const hl = highlightLayer(a, a.render === 'heatmap' ? 5 : pointRadiusOf(a), line);
    if (hl) extra.push(hl);
  }
  if (a.labels?.items.length) extra.push(labelLayer(a, a.render === 'heatmap' ? 5 : pointRadiusOf(a)));
  return [...(Array.isArray(base) ? base : [base]), ...extra];
}
