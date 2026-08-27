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
