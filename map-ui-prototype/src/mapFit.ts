import type { FeatureCollection } from 'geojson';

/** A [minLon, minLat, maxLon, maxLat] box as a FeatureCollection, so anything that frames
 *  features can frame a raster's footprint too. */
export function bboxToFC(b: [number, number, number, number]): FeatureCollection {
  const [w, s, e, n] = b;
  return {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature', properties: {},
      geometry: { type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] },
    }],
  };
}
