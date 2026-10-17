/** How a code run reads in the transcript: what it is for, not its source.
 *
 *  The trace used to print `execute_code({'code': 'import geopandas as gpd\nimport pandas…)` for
 *  every run, twelve times in one T16 turn, which told the reader nothing about which run was
 *  which. The agent now passes `purpose` (a phrase completing "Executing code to …"); older
 *  turns and a model that leaves it out fall back to the run's `label`, then to the program's
 *  own first comment or docstring. */

function sentence(text: string, max = 110): string {
  const t = text.replace(/\s+/g, ' ').trim().replace(/\.$/, '');
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

/** `slr_building_counts` → `slr building counts`. */
export function humanizeSlug(slug: string): string {
  return slug.replace(/[_\-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

/** The first comment or docstring line of a program, if it says something. */
export function firstComment(code: string): string | null {
  for (const raw of String(code || '').split('\n').slice(0, 15)) {
    const line = raw.trim();
    if (!line) continue;
    const doc = line.match(/^(?:"""|''')\s*(.+?)\s*(?:"""|''')?$/);
    if (doc && doc[1]) return doc[1];
    const hash = line.match(/^#\s*(.+)$/);
    if (hash) {
      if (/^!|coding[:=]|^-\*-/.test(hash[1])) continue;   // shebang / encoding lines
      return hash[1];
    }
    if (!/^(import|from)\s/.test(line)) return null;          // past the imports: no header
  }
  return null;
}

/** "Executing code to …" for an execute_code call's arguments. */
export function codeRunPhrase(args: any): string {
  const a = (args && typeof args === 'object') ? args : {};
  const purpose = typeof a.purpose === 'string' ? a.purpose.trim().replace(/^to\s+/i, '') : '';
  // "Count hotels…" reads as "Executing code to Count hotels…"; lower the first letter unless
  // it starts an acronym ("OSM", "CRS").
  if (purpose) return `Executing code to ${sentence(/^[A-Z][a-z]/.test(purpose) ? purpose[0].toLowerCase() + purpose.slice(1) : purpose)}`;
  if (typeof a.label === 'string' && a.label.trim()) return `Executing code: ${sentence(humanizeSlug(a.label))}`;
  const comment = typeof a.code === 'string' ? firstComment(a.code) : null;
  if (comment) return `Executing code: ${sentence(comment)}`;
  if (typeof a.entrypoint === 'string' && a.entrypoint.trim()) return `Running ${a.entrypoint.trim()}`;
  return 'Executing code';
}

/** The run bar's title, from a code_progress event's purpose / label. */
export function runBarTitle(p: { purpose?: string; label?: string }): string {
  if (p.purpose) return codeRunPhrase({ purpose: p.purpose });
  const label = p.label && p.label !== 'code' ? humanizeSlug(p.label) : '';
  return label ? `Running code · ${label}` : 'Running code';
}
