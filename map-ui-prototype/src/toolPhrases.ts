/** What a tool call is doing, in words: the "detailed" trace.
 *
 *  The trace printed every call as `inspect_vector({'file_id': 'file_885af7d38fd5'})`, which says
 *  which function ran and nothing a reader can use: which file, what for. Each tool here gets a
 *  phrase built from its own arguments ("Inspecting buildings.geojson", "Searching OpenStreetMap
 *  for hotels in Waikīkī"). The raw call is kept on the trace line and shown in debug mode.
 *  A tool missing from the table still reads as words: its name, humanised, with the most
 *  telling argument.
 *
 *  Argument names are the tools' own (agent_runtime/*_tools.py), read 2026-10-17. */
import { codeRunPhrase } from './codeRun';

export type FileNamer = (fileId: string) => string | undefined;

type Phrase = (a: Record<string, any>, f: (id: any) => string) => string;

function str(v: any, max = 80): string {
  if (v === undefined || v === null) return '';
  const s = (Array.isArray(v) ? v.join(', ') : typeof v === 'object' ? JSON.stringify(v) : String(v))
    .replace(/\s+/g, ' ').trim();
  return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

function quote(v: any, max = 70): string {
  const s = str(v, max);
  return s ? `“${s}”` : '';
}

function list(v: any, max = 4): string {
  const xs = Array.isArray(v) ? v : v ? [v] : [];
  const shown = xs.slice(0, max).map((x) => str(typeof x === 'object' ? (x.name ?? x.place ?? x) : x, 40));
  return shown.join(', ') + (xs.length > max ? ` and ${xs.length - max} more` : '');
}

function where(a: Record<string, any>, f: (id: any) => string): string {
  if (a.file_id) return `for ${f(a.file_id)}`;
  if (a.bbox) return 'for the given area';
  if (a.lat !== undefined && a.lon !== undefined) return `around ${Number(a.lat).toFixed(3)}, ${Number(a.lon).toFixed(3)}`;
  return '';
}

function join(...parts: string[]): string {
  return parts.filter(Boolean).join(' ');
}

const PHRASES: Record<string, Phrase> = {
  execute_code: (a) => codeRunPhrase(a),
  write_workspace_file: (a) => join('Writing', str(a.path)),
  read_workspace_file: (a) => join('Reading', str(a.path)),
  edit_workspace_file: (a) => join('Editing', str(a.path)),
  write_output_file: (a) => join('Saving', str(a.filename)),
  write_text_file: (a) => join('Writing', str(a.path)),
  read_text_file: (a) => join('Reading', str(a.path)),
  inspect_file_for_analysis: (a) => join('Reading', str(a.path)),

  // Searching
  web_search: (a) => join('Searching the web for', quote(a.query)),
  web_fetch: (a) => join('Reading', str(a.url, 70)),
  agent_kb_search: (a) => join('Searching the I-GUIDE knowledge base for', quote(a.query)),
  kb_method_search: (a) => join('Looking up methods for', quote(a.query)),
  keyword_search: (a) => join('Searching the platform for', quote(a.query)),
  semantic_search: (a) => join('Searching the platform for', quote(a.query)),
  spatial_search: (a) => join('Searching the platform by location for', quote(a.query)),
  opengeodata_search: (a) => join('Searching OpenGeoData for', quote(a.query)),
  neo4j_search: (a) => join('Searching the knowledge graph for', quote(a.query)),
  neo4j_get_element_by_id: (a) => join('Opening platform element', str(a.element_id)),
  neo4j_explore_related_nodes: (a) => join('Finding elements related to', str(a.element_id)),
  get_kb_block: (a) => join('Reading knowledge-base entry', str(a.doc_id)),
  get_method_contract: (a) => join('Reading how to call', str(a.symbol)),
  rerank_evidence: (a) => join('Ranking the evidence for', quote(a.query)),
  list_available_skills: () => 'Listing the available skills',
  load_skill: (a) => join('Loading the skill', quote(a.name)),
  request_capability: (a) => join('Asking for', str(a.capability), a.reason ? `(${str(a.reason, 60)})` : ''),

  // Places and data
  geocode_places: (a) => join('Locating', list(a.places)),
  reverse_geocode: (a) => `Naming the place${Array.isArray(a.points) && a.points.length > 1 ? `s at ${a.points.length} points` : ' at a point'}`,
  overpass_search: (a) => join('Searching OpenStreetMap for', str(a.feature, 50) || 'features',
                               a.place ? `in ${str(a.place, 50)}` : a.bbox ? 'in the given area' : ''),
  admin_boundary: (a) => join('Fetching the boundary of', str(a.area, 50) || 'the area', a.state ? `(${str(a.state, 30)})` : ''),
  fetch_public_data: (a) => join('Downloading', str(a.filename) || str(a.url, 70)),
  stage_url: (a) => join('Fetching', str(a.filename) || str(a.url, 70)),
  stage_element: (a) => join('Fetching platform element', str(a.element_id)),
  list_staged_inputs: () => 'Listing the staged inputs',
  list_conversation_files: () => "Listing this conversation's files",

  // Map
  add_map_layer: (a, f) => join('Adding', f(a.file_id), 'to the map', a.render && a.render !== 'auto' ? `as ${str(a.render)}` : '',
                                a.column ? `by ${str(a.column, 30)}` : ''),
  add_raster_layer: (a, f) => join('Adding', f(a.file_id), 'to the map'),
  render_map_image: (a, f) => join('Drawing a map image of', f(a.file_id), a.column ? `by ${str(a.column, 30)}` : ''),

  // Vector operations
  inspect_vector: (a, f) => join('Inspecting', f(a.file_id)),
  vector_to_geojson: (a, f) => join('Converting', f(a.file_id), 'to GeoJSON'),
  reproject_vector: (a, f) => join('Reprojecting', f(a.file_id), a.target_crs ? `to ${str(a.target_crs)}` : ''),
  vector_spatial_join: (a, f) => join('Joining', f(a.left_file_id), 'with', f(a.right_file_id),
                                      a.predicate ? `(${str(a.predicate)})` : ''),
  clip_layer: (a, f) => join('Clipping', f(a.target_file_id), 'to', f(a.clip_file_id)),
  dissolve_layer: (a, f) => join('Dissolving', f(a.file_id), a.by ? `by ${str(a.by, 30)}` : ''),
  intersect_layers: (a, f) => join('Intersecting', f(a.left_file_id), 'with', f(a.right_file_id)),
  erase_layer: (a, f) => join('Removing', f(a.erase_file_id), 'from', f(a.target_file_id)),
  buffer_layer: (a, f) => join('Buffering', f(a.file_id), a.distance !== undefined ? `by ${str(a.distance)} ${str(a.units) || ''}`.trim() : ''),
  simplify_layer: (a, f) => join('Simplifying', f(a.file_id)),
  geometry_summary: (a, f) => join('Measuring the shapes in', f(a.file_id)),
  select_by_attribute: (a, f) => join('Selecting from', f(a.file_id),
                                      a.column ? `where ${str(a.column, 30)} ${str(a.op) || ''} ${str(a.value, 30)}`.trim() : ''),
  count_points_in_areas: (a, f) => join('Counting', f(a.points_file_id), 'in', f(a.areas_file_id)),
  aggregate_to_grid: (a, f) => join('Gridding', f(a.points_file_id), a.cell_km ? `into ${str(a.cell_km)} km cells` : ''),
  nearest_distance: (a, f) => join('Measuring the distance from', f(a.from_file_id), 'to the nearest of', f(a.to_file_id)),
  cluster_points: (a, f) => join('Finding clusters in', f(a.file_id)),
  summary_statistics: (a, f) => join('Summarising', f(a.file_id), a.column ? `(${str(a.column, 30)})` : '', a.by ? `by ${str(a.by, 30)}` : ''),

  // Time
  detect_time_column: (a, f) => join('Finding the time column in', f(a.file_id)),
  filter_by_time: (a, f) => join('Filtering', f(a.file_id), a.start || a.end ? `to ${str(a.start) || '…'} – ${str(a.end) || '…'}` : ''),
  time_series: (a, f) => join('Building a time series of', f(a.file_id), a.freq ? `(${str(a.freq)})` : ''),
  compare_periods: (a, f) => join('Comparing two periods of', f(a.file_id)),
  temporal_hotspots: (a, f) => join('Finding emerging hot spots in', f(a.file_id)),

  // Spatial statistics
  global_spatial_autocorrelation: (a, f) => join("Testing", f(a.file_id), a.column ? `(${str(a.column, 30)})` : '', 'for spatial clustering (Moran’s I)'),
  local_moran_lisa: (a, f) => join('Finding local clusters and outliers in', f(a.file_id), a.column ? `(${str(a.column, 30)})` : ''),
  local_getis_ord: (a, f) => join('Finding hot and cold spots in', f(a.file_id), a.column ? `(${str(a.column, 30)})` : ''),
  moran_scatterplot: (a, f) => join('Drawing a Moran scatterplot of', f(a.file_id)),
  spatial_regression: (a, f) => join('Fitting a spatial regression of', str(a.y_column, 30), 'on', list(a.x_columns, 3)),
  regionalize: (a, f) => join('Grouping', f(a.file_id), a.n_regions ? `into ${str(a.n_regions)} regions` : 'into regions'),
  spatial_weights: (a, f) => join('Building neighbour weights for', f(a.file_id)),

  // Satellite embeddings and terrain
  list_embedding_models: () => 'Listing the satellite embedding models',
  list_prediction_heads: () => 'Listing the prediction models',
  list_embedding_packages: () => 'Listing saved embedding packages',
  embed_region: (a, f) => join('Embedding satellite imagery', where(a, f), a.models ? `with ${list(a.models, 3)}` : ''),
  embed_zones: (a, f) => join('Embedding satellite imagery for each zone of', f(a.file_id)),
  segment_region: (a, f) => join('Segmenting satellite imagery', where(a, f)),
  embedding_change: (a, f) => join('Measuring change between', list(a.years, 3) || 'years', where(a, f)),
  compare_regions: () => 'Comparing two regions in satellite embeddings',
  align_embedding_colors: () => 'Aligning embedding colours across images',
  predict_for_region: (a, f) => join('Predicting land cover', where(a, f)),
  predict_from_package: (a, f) => join('Predicting from', f(a.file_id)),
  fit_zone_model: (a, f) => join('Fitting a model of', str(a.label_column, 30) || 'zone labels', 'from', f(a.vectors_csv_file_id)),
  dem_for_region: (a, f) => join('Fetching elevation', where(a, f)),
  zonal_stats_for_raster: (a, f) => join('Summarising', f(a.raster_file_id), 'within', f(a.polygons_file_id)),
  terrain_derivative: (a, f) => join('Computing', str(a.kind) || 'a terrain derivative', 'from', f(a.raster_file_id)),
  inundation_at_level: (a, f) => join('Mapping flooding at', a.level_m !== undefined ? `${str(a.level_m)} m` : 'a water level', 'on', f(a.raster_file_id)),

  // QGIS
  qgis_processing_help: (a) => join('Reading QGIS help for', str(a.algorithm ?? a.algorithm_id ?? '')),
  qgis_processing_run: (a) => join('Running QGIS', str(a.algorithm)),
  qgis_metric_buffer: (a) => join('Buffering in QGIS', a.distance_meters !== undefined ? `by ${str(a.distance_meters)} m` : ''),
  qgis_map_image: () => 'Rendering a map in QGIS',
  pyqgis_layer_summary: (a) => join('Inspecting', str(a.layer_name) || str(a.layer_path), 'in QGIS'),
  audit_answer_grounding: () => 'Checking the answer against the evidence',
};

const TELLING_ARGS = ['query', 'feature', 'place', 'area', 'path', 'url', 'name', 'column', 'model', 'symbol'];

/** A sentence for one tool call. Never the raw arguments. */
export function describeToolCall(name: string, args: any, fileName?: FileNamer): string {
  const a = (args && typeof args === 'object' && !Array.isArray(args)) ? args : {};
  const f = (id: any): string => {
    if (Array.isArray(id)) return list(id.map((x): string => f(x)), 3);
    const key = str(id, 60);
    if (!key) return 'a file';
    return (fileName && fileName(key)) || key;
  };
  const phrase = PHRASES[name];
  if (phrase) {
    try { return phrase(a, f).replace(/\s+/g, ' ').trim(); } catch { /* fall through */ }
  }
  // A tool not in the table (an MCP tool, one added later): its name in words, and the one
  // argument most likely to say what it is about.
  const words = name.replace(/^mcp__\w+?__/, '').replace(/[_\-]+/g, ' ').trim();
  const lead = words.charAt(0).toUpperCase() + words.slice(1);
  const key = TELLING_ARGS.find((k) => a[k] !== undefined && a[k] !== '');
  const fileKey = Object.keys(a).find((k) => /file_id$/.test(k) && a[k]);
  const object = key ? quote(a[key]) : fileKey ? f(a[fileKey]) : '';
  return join(lead, object ? `· ${object}` : '');
}
