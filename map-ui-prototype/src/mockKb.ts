// Mock I-GUIDE Knowledge Base: stands in for the OpenSearch spatial index while
// no live cluster is reachable. Each element carries GeoJSON spatial fields in
// exactly the shape produced by the fixed reindex_wkt_spatial.py
// (spatial-centroid = geo_point, spatial-bounding-box = geo_shape Polygon).
//
// searchKb() mimics the production query documented in Contract 2:
//   { bool: { must: [ knn contents-embedding ],
//             filter: [ geo_shape spatial-bounding-box intersects <drawn> ] } }
// Here: keyword score stands in for kNN; turf.booleanIntersects stands in for
// the geo_shape filter.
import * as turf from '@turf/turf';
import type { Feature, FeatureCollection, Polygon, Geometry } from 'geojson';
import type { SpatialFilter } from './contracts';

export interface KbElement {
  id: string;
  title: string;
  type: 'dataset' | 'notebook' | 'publication';
  contents: string;
  'spatial-centroid': { type: 'Point'; coordinates: [number, number] };
  'spatial-bounding-box': Polygon;
}

function bbox(minLon: number, minLat: number, maxLon: number, maxLat: number): Polygon {
  return {
    type: 'Polygon',
    coordinates: [[
      [minLon, minLat], [maxLon, minLat], [maxLon, maxLat],
      [minLon, maxLat], [minLon, minLat],
    ]],
  };
}
function centroid(lon: number, lat: number): { type: 'Point'; coordinates: [number, number] } {
  return { type: 'Point', coordinates: [lon, lat] };
}

export const KB: KbElement[] = [
  {
    id: 'kb-1', title: 'Illinois Cropland Data Layer 2023', type: 'dataset',
    contents: 'agriculture crop yield corn soybean land cover raster illinois midwest',
    'spatial-centroid': centroid(-89.2, 40.0),
    'spatial-bounding-box': bbox(-91.5, 37.0, -87.0, 42.5),
  },
  {
    id: 'kb-2', title: 'Chicago Urban Heat Island Analysis', type: 'notebook',
    contents: 'urban heat island temperature climate city chicago landsat thermal',
    'spatial-centroid': centroid(-87.63, 41.88),
    'spatial-bounding-box': bbox(-88.0, 41.6, -87.5, 42.05),
  },
  {
    id: 'kb-3', title: 'Mississippi River Basin Flood Risk', type: 'dataset',
    contents: 'flood risk hydrology river basin water inundation mississippi elevation',
    'spatial-centroid': centroid(-90.2, 38.6),
    'spatial-bounding-box': bbox(-95.0, 29.0, -88.0, 47.0),
  },
  {
    id: 'kb-4', title: 'Great Lakes Water Quality Timeseries', type: 'dataset',
    contents: 'water quality lakes nutrient phosphorus algae monitoring great lakes',
    'spatial-centroid': centroid(-84.5, 44.0),
    'spatial-bounding-box': bbox(-92.0, 41.0, -76.0, 49.0),
  },
  {
    id: 'kb-5', title: 'Urbana-Champaign Land Use Change', type: 'notebook',
    contents: 'land use change urban growth sprawl parcels zoning champaign urbana illinois',
    'spatial-centroid': centroid(-88.24, 40.11),
    'spatial-bounding-box': bbox(-88.35, 40.02, -88.15, 40.18),
  },
  {
    id: 'kb-6', title: 'California Wildfire Burn Severity', type: 'dataset',
    contents: 'wildfire fire burn severity vegetation california forest drought',
    'spatial-centroid': centroid(-120.5, 38.5),
    'spatial-bounding-box': bbox(-124.0, 36.0, -118.0, 41.0),
  },
  {
    id: 'kb-7', title: 'US County Population Density 2020', type: 'dataset',
    contents: 'population density census demographics county united states people',
    'spatial-centroid': centroid(-98.0, 39.5),
    'spatial-bounding-box': bbox(-125.0, 24.0, -66.0, 49.5),
  },
  {
    id: 'kb-8', title: 'Iowa Soil Moisture & Drought Index', type: 'publication',
    contents: 'soil moisture drought index agriculture precipitation iowa midwest',
    'spatial-centroid': centroid(-93.5, 42.0),
    'spatial-bounding-box': bbox(-96.6, 40.4, -90.1, 43.5),
  },
];

export interface KbHit extends KbElement {
  score: number;
}

function keywordScore(query: string, el: KbElement): number {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return 0.5;
  const hay = (el.title + ' ' + el.contents).toLowerCase();
  let hits = 0;
  for (const t of terms) if (hay.includes(t)) hits++;
  return hits / terms.length;
}

export function searchKb(query: string, filter?: SpatialFilter | null): KbHit[] {
  let elems = KB;
  if (filter) {
    const region = turf.feature(filter.geometry as Geometry);
    elems = elems.filter((el) => {
      const shape = turf.feature(el['spatial-bounding-box']);
      try {
        if (filter.relation === 'within') return turf.booleanWithin(shape, region);
        if (filter.relation === 'contains') return turf.booleanContains(shape, region);
        return turf.booleanIntersects(shape, region);
      } catch {
        return false;
      }
    });
  }
  return elems
    .map((el) => ({ ...el, score: keywordScore(query, el) }))
    .filter((h) => h.score > 0)
    .sort((a, b) => b.score - a.score);
}

// Render hits as two artifacts' worth of features: centroids (points) + bboxes.
export function kbHitsToFeatureCollections(hits: KbHit[]): {
  centroids: FeatureCollection;
  boxes: FeatureCollection;
} {
  const centroids: Feature[] = hits.map((h) => ({
    type: 'Feature',
    geometry: h['spatial-centroid'],
    properties: { id: h.id, name: h.title, type: h.type, score: h.score.toFixed(2) },
  }));
  const boxes: Feature[] = hits.map((h) => ({
    type: 'Feature',
    geometry: h['spatial-bounding-box'],
    properties: { id: h.id, name: h.title, type: h.type },
  }));
  return {
    centroids: { type: 'FeatureCollection', features: centroids },
    boxes: { type: 'FeatureCollection', features: boxes },
  };
}
