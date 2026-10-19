// The agent <-> map contract (Contract 1 from the design).
// Every visual on the map -- KB search results, Overpass results, user uploads,
// and analysis outputs -- is expressed as a LayerArtifact and rendered by one
// pure function (toDeckLayer). In production the agent emits these instead of
// baked PNGs; here the browser modules produce them locally.

import type { FeatureCollection, Geometry } from 'geojson';

export type RGBA = [number, number, number, number];

export interface Legend {
  label: string;
  // Set only where the colour means something (a hot spot is red) or the user asked for it;
  // every other class takes the map's palette (palette.ts resolveLegend).
  color?: RGBA;
}

// What the agent chose to show of a layer's features (add_map_layer, Stage 80). Named options
// only, validated on the server against the data; the tool result states what they draw, so the
// answer can describe the map rather than the request. No free-form deck.gl style comes this way.
export interface SizeBy {
  column: string;
  min: number;                 // the column's range in the layer as sent
  max: number;
  range: [number, number];     // pixels: radius = lo + (hi - lo) * sqrt(t)
}

export interface Highlight {
  column: string;
  op: 'eq' | 'in' | 'gte' | 'lte' | 'gt' | 'lt';
  value: unknown;
  numeric: boolean;            // compare as numbers (numeric column) or as text
  label: string;               // the legend entry
  color: RGBA;
  count?: number;              // how many features it picks out, as the server counted
}

export interface Labels {
  column: string;
  scope: 'all' | 'highlighted';
  items: { position: [number, number]; text: string }[];   // at most 50, placed by the server
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
      // Pointer to the saved vectors behind the picture, when there are any. Persisted with the
      // layer, so a session restored tomorrow still knows which package each raster came from.
      embedding?: {
        file_id: string;
        filename?: string;
        model?: string;
        months?: string;
        models_in_package?: string[];
        recoloured_on_shared_basis?: boolean;
      };
    }
  | {
      kind: 'geojson';
      id: string;
      source: string;            // provenance: 'kb' | 'overpass' | 'upload' | 'analysis'
      label: string;
      data: FeatureCollection;
      style?: DeckStyle;
      render?: 'geojson' | 'heatmap' | 'points' | 'categories' | 'network' | 'route';
      // 'network' and 'route' draw a road network and a path found on it the way a navigation
      // app does: the network as muted lines beneath everything, the route as a thick cased
      // line on top with its start and end marked (toDeckLayer, AgentMap's layer order).
      // 'categories' shades by a CLASS NAME (LISA's High-High, a Gi* hot-spot band, a region
      // id) using `legend` as the palette, instead of ramping a number. The two cannot share a
      // path: Number('High-High') is NaN, so a categorical column on the choropleth ramp
      // renders every feature in one flat colour.
      styleBy?: string;   // property to shade by: numeric (choropleth) or class name (categories)
      // Draw the boundary only, leaving the interior clear. A filled polygon over a raster
      // hides the raster: the embedded tract came back as a solid violet slab covering the
      // very pixel image it was framing.
      outline?: boolean;
      // How see-through the fill is (0..1), and whether the layer draws beneath the others.
      // The agent sets both for a layer that frames other results: a buffer drawn solid over
      // the schools inside it hid them and the streets beneath (live, 2026-10-18).
      fillOpacity?: number;
      beneath?: boolean;
      partial?: { shown: number; total: number };  // layer is a SAMPLE — shown in the UI
      legend?: Legend[];
      // One colour for the whole layer, set by the agent only when the user asked for it ("the
      // crimes in red"). It wins over the source colour, the route blue and the heatmap ramp.
      color?: RGBA;
      // Per-point size by a column, or one size in pixels; without either, every point is one
      // density-dependent size (densityRadius).
      sizeBy?: SizeBy;
      pointSize?: number;
      // Features matching one predicate, drawn larger, on top, in a stand-out colour.
      highlight?: Highlight;
      labels?: Labels;
      lineWidth?: number;          // pixels
      lineColor?: RGBA;
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
