// Port of the reference chat prototype's evidence/sources extraction so the map UI
// renders the same "Sources used" block under an answer.
export interface SourceDoc {
  doc_id?: string; id?: string; url?: string; title?: string;
  element_type?: string; 'resource-type'?: string; source?: string; source_system?: string;
  provider?: string; license?: string; abstract?: string; snippet?: string; contents?: string;
  bbox?: number[];
}
export type SourceGroup = 'internal' | 'external' | 'web';

export function evidenceDocs(response: any): SourceDoc[] {
  const out: SourceDoc[] = [];
  const seen = new Set<string>();
  const add = (d: any) => {
    if (!d || typeof d !== 'object') return;
    const src = (d.document && typeof d.document === 'object') ? d.document : d;
    const id = String(src.doc_id || src.id || src.url || src.title || '');
    if (!id || seen.has(id)) return;
    seen.add(id); out.push(src);
  };
  const orch = response?.agent_result?.orchestration_result;
  if (orch && Array.isArray(orch.evidence)) orch.evidence.forEach(add);
  if (Array.isArray(response?.opengeodata_results)) response.opengeodata_results.forEach(add);
  // Also accept the simplified `elements` array the API returns at top level.
  if (Array.isArray(response?.elements)) response.elements.forEach(add);
  return out;
}

export function sourceGroup(src: SourceDoc): SourceGroup {
  const et = String(src.element_type || src['resource-type'] || '').toLowerCase();
  const sn = String(src.source || src.source_system || '').toLowerCase();
  if (et === 'web' || sn === 'web') return 'web';
  if (et === 'opengeodata' || sn === 'opengeodata') return 'external';
  return 'internal';
}

export function groupedSources(response: any): Record<SourceGroup, SourceDoc[]> {
  const groups: Record<SourceGroup, SourceDoc[]> = { internal: [], external: [], web: [] };
  for (const d of evidenceDocs(response)) groups[sourceGroup(d)].push(d);
  return groups;
}

// --- where a source links to ------------------------------------------------------------
// The I-GUIDE platform. Not configurable here because the client has no FRONTEND_DOMAIN; if a
// deployment moves, this is the one line to change.
const PLATFORM = 'https://platform.i-guide.io';

// Element types that really ARE pages on the platform, and the path each lives under.
// An ALLOWLIST, deliberately: the server once derived these paths by pluralising whatever
// element_type it was handed, which turned an OpenStreetMap hit into
// platform.i-guide.io/osm_features/osm:node:767555934 — a 404 that looked like a real citation.
// A type that is not on this list gets NO link rather than an invented one.
const PLATFORM_PATHS: Record<string, string> = {
  dataset: 'datasets',
  notebook: 'notebooks',
  publication: 'publications',
  oer: 'oers',
  map: 'maps',
  code: 'code',          // singular on the platform, unlike the rest
};

/** The href for a source item, or '' when no honest link can be formed.
 *
 * URL-FIRST: anything carrying its own url is cited there, whatever its type — external hits
 * (OpenGeoData, OpenStreetMap, open web) always do. Internal knowledge elements arrive with
 * url '' and are linked by type + doc_id instead. The server fills this in too; doing it here
 * as well means the sources list is clickable regardless of which backend answered.
 */
export function sourceHref(src: SourceDoc): string {
  const url = String(src.url || '').trim();
  if (/^https?:\/\//i.test(url)) return url;
  const etype = String(src.element_type || src['resource-type'] || '').trim().toLowerCase();
  const id = String(src.doc_id || src.id || '').trim();
  const path = PLATFORM_PATHS[etype];
  if (!path || !id) return '';
  return `${PLATFORM}/${path}/${encodeURIComponent(id)}`;
}
