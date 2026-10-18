// Run with `npm run check:coderun`. How a code run reads in the transcript.
import { codeRunPhrase, firstComment, runBarTitle } from './codeRun';
import { describeToolCall } from './toolPhrases';

let failed = 0;
function eq(name: string, got: unknown, want: unknown) {
  if (got !== want) { failed++; console.error(`FAIL ${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); }
}

eq('purpose', codeRunPhrase({ purpose: 'count buildings inside the sea-level-rise area', code: 'import x' }),
   'Executing code to count buildings inside the sea-level-rise area');
eq('purpose with a leading "to" and a full stop', codeRunPhrase({ purpose: 'To snap blocks to the street network.' }),
   'Executing code to snap blocks to the street network');
eq('label when no purpose', codeRunPhrase({ label: 'slr_building_counts', code: 'import x' }),
   'Executing code: slr building counts');
eq('first comment when no label', codeRunPhrase({ code: 'import geopandas as gpd\n# Walk each block out of the zone\nx = 1' }),
   'Executing code: Walk each block out of the zone');
eq('docstring', firstComment('"""Count hotels in the zone."""\nimport x'), 'Count hotels in the zone.');
eq('shebang is not a description', firstComment('#!/usr/bin/env python\nimport x\nx = 1'), null);
eq('code with no header', codeRunPhrase({ code: 'import geopandas as gpd\nimport pandas as pd\nb = 1' }), 'Executing code');
eq('entrypoint', codeRunPhrase({ entrypoint: 'main.py' }), 'Running main.py');
eq('never the raw source', codeRunPhrase({ code: "import geopandas as gpd\nimport pandas as pd" }).includes('import'), false);
eq('long purposes are clipped', codeRunPhrase({ purpose: 'x'.repeat(300) }).length <= 'Executing code to '.length + 110, true);
eq('run bar with purpose', runBarTitle({ purpose: 'count hotels', label: 'hotels' }), 'Executing code to count hotels');
eq('run bar with label only', runBarTitle({ label: 'slr_building_counts' }), 'Running code · slr building counts');
eq('run bar default', runBarTitle({ label: 'code' }), 'Running code');

// Every tool call reads as words (toolPhrases).
const names = (id: string) => ({ file_885af7d38fd5: 'buildings.geojson', file_2437c7f294d7: 'slr_exposure_3_2ft.geojson' } as Record<string, string>)[id];
eq('inspect_vector names the file', describeToolCall('inspect_vector', { file_id: 'file_885af7d38fd5' }, names), 'Inspecting buildings.geojson');
eq('an unknown id stays an id', describeToolCall('inspect_vector', { file_id: 'file_000000000000' }, names), 'Inspecting file_000000000000');
eq('overpass', describeToolCall('overpass_search', { feature: 'hotels', place: 'Waikīkī' }), 'Searching OpenStreetMap for hotels in Waikīkī');
eq('join', describeToolCall('vector_spatial_join', { left_file_id: 'file_885af7d38fd5', right_file_id: 'file_2437c7f294d7', predicate: 'intersects' }, names),
   'Joining buildings.geojson with slr_exposure_3_2ft.geojson (intersects)');
eq('web search', describeToolCall('web_search', { query: 'shapely prepare' }), 'Searching the web for “shapely prepare”');
eq('execute_code goes through codeRun', describeToolCall('execute_code', { purpose: 'count hotels' }), 'Executing code to count hotels');
eq('a tool not in the table', describeToolCall('mcp__platform__find_datasets', { query: 'flood' }), 'Find datasets · “flood”');
eq('a tool with no arguments', describeToolCall('list_embedding_models', {}), 'Listing the satellite embedding models');
for (const tool of ['add_map_layer', 'buffer_layer', 'local_getis_ord', 'embed_region', 'reverse_geocode', 'geocode_places',
                    'select_by_attribute', 'filter_by_time', 'count_points_in_areas', 'qgis_processing_run']) {
  const s = describeToolCall(tool, { file_id: 'file_885af7d38fd5', places: ['Honolulu'], points: [[21.3, -157.8]] }, names);
  eq(`${tool} reads as words`, /[({]|_/.test(s.replace('buildings.geojson', '')), false);
}

eq('a capitalised purpose reads as a clause', codeRunPhrase({ purpose: 'Count hotels in the zone' }), 'Executing code to count hotels in the zone');
eq('an acronym keeps its case', codeRunPhrase({ purpose: 'OSM streets to a graph' }), 'Executing code to OSM streets to a graph');

if (failed) { console.error(`${failed} code-run check(s) failed`); process.exit(1); }
console.log('code-run phrasing: all checks pass');
