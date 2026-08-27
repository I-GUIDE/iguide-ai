// Live Overpass API integration: query OSM features inside a bbox.
// Returns real geometries (Point / LineString / Polygon) via `out geom`, so
// linear features like rivers render as lines, not center dots.
import type { FeatureCollection, Feature, Geometry } from 'geojson';

const ENDPOINT = 'https://overpass-api.de/api/interpreter';

export interface OverpassPreset {
  key: string;
  label: string;
  filter: string;         // Overpass tag filter, e.g. 'amenity=cafe'
  kinds?: ('node' | 'way' | 'relation')[]; // which element kinds to fetch
}

export const OVERPASS_PRESETS: OverpassPreset[] = [
  { key: 'cafe', label: 'Cafés', filter: 'amenity=cafe', kinds: ['node', 'way'] },
  { key: 'restaurant', label: 'Restaurants', filter: 'amenity=restaurant', kinds: ['node', 'way'] },
  { key: 'school', label: 'Schools', filter: 'amenity=school', kinds: ['node', 'way'] },
  { key: 'hospital', label: 'Hospitals', filter: 'amenity=hospital', kinds: ['node', 'way'] },
  { key: 'park', label: 'Parks', filter: 'leisure=park', kinds: ['way'] },
  { key: 'supermarket', label: 'Supermarkets', filter: 'shop=supermarket', kinds: ['node', 'way'] },
  { key: 'river', label: 'Rivers', filter: 'waterway=river', kinds: ['way'] },
  { key: 'stream', label: 'Streams', filter: 'waterway=stream', kinds: ['way'] },
  { key: 'water', label: 'Water bodies', filter: 'natural=water', kinds: ['way'] },
  { key: 'road', label: 'Major roads', filter: 'highway=primary', kinds: ['way'] },
];

function wayToGeometry(el: any): Geometry | null {
  const g = el.geometry;
  if (!Array.isArray(g) || g.length < 2) return null;
  const coords = g.map((p: any) => [p.lon, p.lat]);
  const closed =
    coords.length > 3 &&
    coords[0][0] === coords[coords.length - 1][0] &&
    coords[0][1] === coords[coords.length - 1][1];
  const areaLike = el.tags && (el.tags.natural || el.tags.landuse || el.tags.leisure || el.tags.building || el.tags.amenity);
  if (closed && areaLike) return { type: 'Polygon', coordinates: [coords] };
  return { type: 'LineString', coordinates: coords };
}

// bbox as [minLon, minLat, maxLon, maxLat]
export async function queryOverpass(
  preset: OverpassPreset,
  bbox: [number, number, number, number],
  signal?: AbortSignal,
): Promise<FeatureCollection> {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const b = `${minLat},${minLon},${maxLat},${maxLon}`; // Overpass = S,W,N,E
  const kinds = preset.kinds ?? ['node', 'way'];
  const body = kinds.map((k) => `${k}[${preset.filter}](${b});`).join('\n      ');
  const q = `
    [out:json][timeout:25];
    (
      ${body}
    );
    out geom 800;`;

  const res = await fetch(ENDPOINT, {
    method: 'POST',
    body: 'data=' + encodeURIComponent(q),
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    signal,
  });
  if (!res.ok) throw new Error(`Overpass HTTP ${res.status}`);
  const json = await res.json();

  const features: Feature[] = [];
  for (const el of json.elements ?? []) {
    let geom: Geometry | null = null;
    if (el.type === 'node' && el.lon != null) {
      geom = { type: 'Point', coordinates: [el.lon, el.lat] };
    } else if (el.type === 'way') {
      geom = wayToGeometry(el);
      if (!geom && el.center) geom = { type: 'Point', coordinates: [el.center.lon, el.center.lat] };
    }
    if (!geom) continue;
    features.push({
      type: 'Feature',
      geometry: geom,
      properties: { name: el.tags?.name ?? '(unnamed)', osm_id: el.id, ...el.tags },
    });
  }
  return { type: 'FeatureCollection', features };
}
