// The thin renderer: LayerArtifact -> deck.gl layer. One switch, easy to test.
import { BitmapLayer, GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import type { LayerArtifact } from './contracts';
import type { Feature, Point } from 'geojson';
import { routeEnds } from './routeGeometry';

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

// Categorical fill: look the feature's class up in the legend the tool sent. The palette is
// NOT hardcoded here on purpose -- the tool that assigned the classes is the only thing that
// knows what they mean, so a new categorical statistic needs no change in this file.
function categoricalFor(a: VectorLayer): ((f: any) => [number, number, number, number]) | null {
  if (!a.styleBy || !a.legend?.length) return null;
  const key = a.styleBy;
  const byLabel = new Map(a.legend.map((e) => [e.label, e.color]));
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
  const layers: any[] = [
    new GeoJsonLayer({ id: `${a.id}--casing`, data: a.data, visible, pickable: false, ...line,
                       getLineColor: ROUTE_CASING, getLineWidth: 11 }),
    new GeoJsonLayer({ id: a.id, data: a.data, visible, pickable: true, ...line,
                       getLineColor: ROUTE_LINE, getLineWidth: 6 }),
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
    lineJointRounded: true, getLineColor: NETWORK_LINE, getLineWidth: 1.5,
    pointType: 'circle', getPointRadius: 2, pointRadiusUnits: 'pixels',
  });
}

export function toDeckLayer(a: LayerArtifact) {
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
  const fill = s.fill ?? DEFAULT_FILL;
  const line = s.line ?? DEFAULT_LINE;

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
    getFillColor: ramp ?? fill,
    // On a choropleth the FILL carries the number, so the outline must stay out of its way:
    // a 708-cell hex grid drawn with the standard 2px purple border read as an empty mesh —
    // the borders covered more pixels than the shaded interiors. Hairline grey instead.
    getLineColor: a.outline ? (s.line ?? [90, 60, 160, 255]) : (ramp ? [90, 90, 105, 90] : line),
    getLineWidth: a.outline ? (s.lineWidth ?? 2.5)
                            : (ramp ? (s.lineWidth ?? 0.5) : (s.lineWidth ?? 2)),
    lineWidthUnits: 'pixels',
    getPointRadius: s.pointRadius ?? 6,
    pointRadiusUnits: s.radiusUnits ?? 'pixels',
    opacity: s.opacity ?? 1,
    // expose source + props for the tooltip
    updateTriggers: { getFillColor: [a.styleBy, a.data, a.render, a.legend] },
  });
}
