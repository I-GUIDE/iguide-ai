import { useCallback, useEffect, useMemo, useRef } from 'react';
import { Map, Source, Layer, useControl } from 'react-map-gl/maplibre';
import type { MapRef } from 'react-map-gl/maplibre';
import type { MapLayerMouseEvent } from 'react-map-gl/maplibre';
import { MapboxOverlay } from '@deck.gl/mapbox';
import type { Feature, Polygon } from 'geojson';
import type { LayerArtifact } from '../contracts';
import { toDeckLayer } from '../toDeckLayer';
import 'maplibre-gl/dist/maplibre-gl.css';

// Raster OSM basemap -- no API key required (fine for a prototype).
const OSM_STYLE: any = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
};

function getTooltip({ object }: any) {
  if (!object) return null;
  const p = object.properties ?? {};
  const name = p.name ?? p.id ?? object.id ?? 'feature';
  const extra = p.type ? ` (${p.type})` : p.amenity ? ` (${p.amenity})` : '';
  const score = p.score ? `  score ${p.score}` : '';
  return { text: `${name}${extra}${score}` };
}

function DeckOverlay({ layers, onFeatureClick }: { layers: any[]; onFeatureClick: (feature: any, layerId: string) => void }) {
  const onClick = (info: any) => { if (info && info.object) onFeatureClick(info.object, info.layer?.id ?? ''); };
  const overlay = useControl(() => new MapboxOverlay({ interleaved: true, layers, getTooltip, onClick }));
  (overlay as MapboxOverlay).setProps({ layers, getTooltip, onClick });
  return null;
}

interface Props {
  layers: LayerArtifact[];
  drawnRegion: Polygon | null;
  drawPreview: Feature | null;
  onMapClick: (lng: number, lat: number) => void;
  onHover: (info: any) => void;
  onFeatureClick: (feature: any, layerId: string) => void;
  onReady?: () => void;
  onResize?: () => void;
  /** Live rectangle while the right button is held (null clears it). */
  onRegionPreview?: (poly: Polygon | null) => void;
  /** Final selection: a right-DRAG box, or a right-CLICK expanded to `clickBoxKm`. */
  onRegionDrawn?: (poly: Polygon, viaClick: boolean) => void;
}

/** Axis-aligned box from two lng/lat corners. */
function boxFrom(a: [number, number], b: [number, number]): Polygon {
  const [w, e] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [s, n] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  return { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] };
}

// A right-click that barely moved is a click, not a drag — same 8px threshold the rs-embed
// demo uses — and becomes a box of this size so a single click still selects something.
const CLICK_PX = 8;
const CLICK_BOX_KM = 2;

export function AgentMap({ layers, drawnRegion, drawPreview, onMapClick, onHover, onFeatureClick, onReady, onResize, onRegionPreview, onRegionDrawn }: Props) {
  // Right-drag selects a region (matching the rs-embed demo). Wired straight onto the
  // MapLibre instance rather than through React handlers because it needs `mousedown`/
  // `mouseup` with the BUTTON number, and because dragRotate has to be turned off first:
  // MapLibre rotates the map on right-drag by default, so the map would spin while drawing.
  const dragRef = useRef<{ start: [number, number]; startPx: [number, number] } | null>(null);
  const cbRef = useRef({ onRegionPreview, onRegionDrawn });
  cbRef.current = { onRegionPreview, onRegionDrawn };
  // Bound from the `ref` callback below, the moment the instance is published — an effect
  // would either miss it (empty deps, instance not yet published) or re-bind on every
  // render. `unbindRef` holds the teardown so unmount can undo it exactly once.
  const boundRef = useRef<any>(null);
  const unbindRef = useRef<(() => void) | null>(null);
  const bindRegionDrag = useCallback((m: any) => {
    if (!m || boundRef.current === m) return;
    boundRef.current = m;
    const container = m.getContainer?.();
    const noMenu = (e: Event) => e.preventDefault();   // else the browser menu covers the box
    container?.addEventListener('contextmenu', noMenu);
    try { m.dragRotate?.disable(); m.touchZoomRotate?.disableRotation?.(); } catch { /* */ }

    const down = (e: any) => {
      if (e.originalEvent?.button !== 2) return;
      dragRef.current = { start: [e.lngLat.lng, e.lngLat.lat], startPx: [e.point.x, e.point.y] };
      cbRef.current.onRegionPreview?.(null);
    };
    const move = (e: any) => {
      const d = dragRef.current;
      if (!d) return;
      cbRef.current.onRegionPreview?.(boxFrom(d.start, [e.lngLat.lng, e.lngLat.lat]));
    };
    const up = (e: any) => {
      const d = dragRef.current;
      if (!d) return;
      dragRef.current = null;
      cbRef.current.onRegionPreview?.(null);
      const moved = Math.hypot(e.point.x - d.startPx[0], e.point.y - d.startPx[1]);
      if (moved < CLICK_PX) {
        // Right-click: a box of CLICK_BOX_KM centred on the point. Longitude degrees
        // shrink with latitude, so scale them by cos(lat) or the box is wide near the poles.
        const [lng, lat] = d.start;
        const dLat = CLICK_BOX_KM / 2 / 111.32;
        const dLng = dLat / Math.max(Math.cos((lat * Math.PI) / 180), 0.01);
        cbRef.current.onRegionDrawn?.(boxFrom([lng - dLng, lat - dLat], [lng + dLng, lat + dLat]), true);
      } else {
        cbRef.current.onRegionDrawn?.(boxFrom(d.start, [e.lngLat.lng, e.lngLat.lat]), false);
      }
    };
    m.on('mousedown', down); m.on('mousemove', move); m.on('mouseup', up);
    unbindRef.current = () => {
      container?.removeEventListener('contextmenu', noMenu);
      try { m.off('mousedown', down); m.off('mousemove', move); m.off('mouseup', up); } catch { /* */ }
      boundRef.current = null;
    };
  }, []);
  useEffect(() => () => { unbindRef.current?.(); unbindRef.current = null; }, []);

  // The map is mounted while hidden (progressive reveal), so its canvas is sized for a
  // zero/40x30 box and stays that way: observed 400x300 inside an 820x646 container, painting
  // nothing. A one-shot resize on reveal races the layout, so track the container instead.
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapRef | null>(null);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      const m = (window as any).__map;
      if (m && el.clientWidth > 0) {
        try { m.resize(); } catch { /* */ }
        // Resizing keeps the CENTER but not the framing: a fit computed against the
        // pre-reveal box stays over-zoomed after the canvas grows, so let the owner
        // re-apply it now that the container is its real size.
        onResize?.();
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [onResize]);

  const deckLayers = useMemo(
    () => layers.map((a) => toDeckLayer(a)),
    [layers],
  );

  const regionFeature: Feature | null = useMemo(() => {
    if (!drawnRegion) return null;
    return { type: 'Feature', geometry: drawnRegion, properties: {} };
  }, [drawnRegion]);

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0 }}>
    <Map
      initialViewState={{ longitude: -89.0, latitude: 40.5, zoom: 5.2 }}
      mapStyle={OSM_STYLE}
      // Keep the WebGL buffer so the rendered view can be exported as an image;
      // without it canvas.toDataURL() returns a cleared, single-colour frame.
      preserveDrawingBuffer
      cursor="grab"
      onClick={(e: MapLayerMouseEvent) => onMapClick(e.lngLat.lng, e.lngLat.lat)}
      ref={(r) => {
        // Publish the instance as soon as it EXISTS. Waiting for the `load` event is
        // unreliable: if the basemap's tile source fails (throttled/blocked), `load` never
        // fires, `__map` stays unset and everything keyed off it — auto-framing a delivered
        // layer, resize, export — silently never happens, even though deck.gl is drawing.
        mapRef.current = r;
        const m = r?.getMap?.();
        if (m && (window as any).__map !== m) {
          (window as any).__map = m;
          bindRegionDrag(m);          // right-drag region selection
          const announce = () => onReady?.();
          if (m.isStyleLoaded?.()) announce();
          else { m.once?.('idle', announce); setTimeout(announce, 3000); }
        }
      }}
      onLoad={(e: any) => { (window as any).__map = e.target; bindRegionDrag(e.target); onReady?.(); }}
      interactiveLayerIds={[]}
      onMouseMove={() => {}}
      style={{ position: 'absolute', inset: 0 }}
    >
      <DeckOverlay layers={deckLayers} onFeatureClick={onFeatureClick} />

      {regionFeature && (
        <Source id="drawn-region" type="geojson" data={regionFeature}>
          <Layer id="drawn-fill" type="fill" paint={{ 'fill-color': '#ff7f0e', 'fill-opacity': 0.12 }} />
          <Layer id="drawn-line" type="line" paint={{ 'line-color': '#ff7f0e', 'line-width': 2, 'line-dasharray': [2, 1] }} />
        </Source>
      )}
      {drawPreview && (
        <Source id="draw-preview" type="geojson" data={drawPreview}>
          <Layer id="preview-fill" type="fill" paint={{ 'fill-color': '#ffcc00', 'fill-opacity': 0.12 }} />
          <Layer id="preview-line" type="line" paint={{ 'line-color': '#ff9d00', 'line-width': 2 }} />
        </Source>
      )}
    </Map>
    </div>
  );
}
