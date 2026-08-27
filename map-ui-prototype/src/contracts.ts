// The agent <-> map contract (Contract 1 from the design).
// Every visual on the map -- KB search results, Overpass results, user uploads,
// and analysis outputs -- is expressed as a LayerArtifact and rendered by one
// pure function (toDeckLayer). In production the agent emits these instead of
// baked PNGs; here the browser modules produce them locally.

import type { FeatureCollection, Geometry } from 'geojson';

export type RGBA = [number, number, number, number];

export interface Legend {
  label: string;
  color: RGBA;
}

export interface DeckStyle {
  fill?: RGBA;
  line?: RGBA;
  lineWidth?: number;   // pixels
  pointRadius?: number; // meters (deck.gl radius) or pixels if radiusUnits='pixels'
  radiusUnits?: 'meters' | 'pixels';
  opacity?: number;     // 0..1
  extruded?: boolean;
  elevation?: number;
}

export type LayerArtifact =
  | {
      // A georeferenced image draped over a geographic extent — e.g. the PCA-RGB picture of
      // a remote-sensing embedding, or a k-means segmentation mask. It carries no features:
      // there is nothing to click, and `bounds` is what makes it land in the right place.
      kind: 'raster';
      id: string;
      source: string;
      label: string;
      url: string;                                  // image the client fetches
      bounds: [number, number, number, number];     // [minLon, minLat, maxLon, maxLat]
      opacity?: number;
      legend?: Legend[];
      fitBounds?: boolean;
      visible?: boolean;
    }
  | {
      kind: 'geojson';
      id: string;
      source: string;            // provenance: 'kb' | 'overpass' | 'upload' | 'analysis'
      label: string;
      data: FeatureCollection;
      style?: DeckStyle;
      render?: 'geojson' | 'heatmap' | 'points' | 'categories';
      // 'categories' shades by a CLASS NAME (LISA's High-High, a Gi* hot-spot band, a region
      // id) using `legend` as the palette, instead of ramping a number. The two cannot share a
      // path: Number('High-High') is NaN, so a categorical column on the choropleth ramp
      // renders every feature in one flat colour.
      styleBy?: string;   // property to shade by: numeric (choropleth) or class name (categories)
      partial?: { shown: number; total: number };  // layer is a SAMPLE — shown in the UI
      legend?: Legend[];
      fitBounds?: boolean;
      visible?: boolean;
      // Where `data` was fetched from. Not used to render — it exists so a saved session can
      // restore the layer without storing megabytes of geometry in the browser.
      sourceUrl?: string;
    };

export interface MapUpdate {
  layers: LayerArtifact[];
  camera?: { bounds?: [number, number, number, number]; center?: [number, number]; zoom?: number };
}

// Contract 2: a drawn region becomes a spatial filter for retrieval.
export interface SpatialFilter {
  geometry: Geometry;
  relation: 'intersects' | 'within' | 'contains';
}
