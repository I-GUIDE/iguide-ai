// Faithful port of the reference chat prototype's renderMarkdown
// (examples/iguide_chat_prototype.html): headings, fenced code, tables, blockquotes,
// hr, ordered/unordered lists, inline code/images/links/bold/em. Image & link URLs are
// resolved via `resolveUrl` (host-relative /agent/files/<id>/download -> absolute).
export function renderMarkdown(src: string, resolveUrl: (u: string) => string): string {
  const esc = (s: string) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const inline = (s: string): string => {
    const stash: string[] = [];
    const hold = (html: string) => { stash.push(html); return `${stash.length - 1}`; };
    const emph = (t: string) => t
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+)__/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*(?!\s)([^*]+?)\*/g, '$1<em>$2</em>');
    let out = String(s)
      .replace(/`([^`]+)`/g, (_m, c) => hold(`<code>${c}</code>`))
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (_m, alt, u) => {
        const href = String(u).replace(/^sandbox:/i, '');
        let safe = /^(https?:|data:image\/)/i.test(href) ? href
          : href.charAt(0) === '/' ? resolveUrl(href) : '';
        if (!safe) return '';
        safe = safe.replace(/"/g, '%22');
        const a = String(alt).replace(/"/g, '&quot;');
        return hold(`<img class="md-img" src="${safe}" alt="${a}" loading="lazy">`);
      })
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_m, t, u) => {
        const href = String(u).replace(/^sandbox:/i, '');
        let safe = /^(https?:|mailto:)/i.test(href) ? href
          : href.charAt(0) === '/' ? resolveUrl(href) : '#';
        safe = safe.replace(/"/g, '%22');
        return hold(`<a href="${safe}" target="_blank" rel="noopener noreferrer">${emph(t)}</a>`);
      });
    out = emph(out);
    for (let k = stash.length - 1; k >= 0; k--) out = out.split(`${k}`).join(stash[k]);
    return out;
  };

  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  let html = ''; let i = 0; let para: string[] = [];
  // Lists: one top-level list, an open <li> inside it, and one nested list inside that <li>.
  // A model writes a list three ways the old renderer broke: items separated by blank lines (a
  // "loose" list), items followed by indented sub-bullets, and numbering that does not start at 1.
  // Each of the first two closed the <ol> after every item, so every item became its own one-item
  // list and read "1." (seen on a prod answer, 2026-10-13); the third was lost because the number
  // was discarded. Now a blank line or an indented line keeps the list open, and the first
  // item's number becomes `start`.
  let top: '' | 'ol' | 'ul' = ''; let liOpen = false; let sub: '' | 'ol' | 'ul' = '';
  const closeSub = () => { if (sub) { html += `</${sub}>`; sub = ''; } };
  const closeLists = () => { closeSub(); if (liOpen) { html += '</li>'; liOpen = false; } if (top) { html += `</${top}>`; top = ''; } };
  const listItem = (l: string) => {
    const m = l.match(/^(\s*)(?:([-*+])|(\d+)[.)])\s+(.*)$/);
    return m ? { indent: m[1].replace(/\t/g, '    ').length, kind: (m[2] ? 'ul' : 'ol') as 'ul' | 'ol', n: m[3] ? Number(m[3]) : 1, text: m[4] } : null;
  };
  const openList = (kind: 'ol' | 'ul', n: number) => (kind === 'ol' && n !== 1 ? `<ol start="${n}">` : `<${kind}>`);
  const flush = () => { if (para.length) { html += '<p>' + inline(esc(para.join('\n'))).replace(/\n/g, '<br>') + '</p>'; para = []; } };
  const isSep = (l: string) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(l);
  const cells = (l: string) => l.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) { flush(); closeLists(); i++; const code: string[] = []; while (i < lines.length && !/^```\s*$/.test(lines[i])) { code.push(lines[i]); i++; } i++; html += `<pre><code>${esc(code.join('\n'))}</code></pre>`; continue; }
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flush(); closeLists(); const lv = h[1].length; html += `<h${lv}>${inline(esc(h[2]))}</h${lv}>`; i++; continue; }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { flush(); closeLists(); html += '<hr>'; i++; continue; }
    if (line.includes('|') && i + 1 < lines.length && isSep(lines[i + 1])) {
      flush(); closeLists(); const head = cells(line); i += 2; const rows: string[][] = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') { rows.push(cells(lines[i])); i++; }
      html += '<table><thead><tr>' + head.map((c) => `<th>${inline(esc(c))}</th>`).join('') + '</tr></thead><tbody>'
        + rows.map((r) => '<tr>' + r.map((c) => `<td>${inline(esc(c))}</td>`).join('') + '</tr>').join('') + '</tbody></table>';
      continue;
    }
    const bq = line.match(/^>\s?(.*)$/);
    if (bq) { flush(); closeLists(); const q: string[] = []; while (i < lines.length) { const m = lines[i].match(/^>\s?(.*)$/); if (!m) break; q.push(m[1]); i++; } html += `<blockquote>${inline(esc(q.join('\n'))).replace(/\n/g, '<br>')}</blockquote>`; continue; }
    const item = listItem(line);
    if (item && top && liOpen && item.indent >= 2) {
      // A nested item belongs to the open <li>.
      flush(); if (sub && sub !== item.kind) closeSub();
      if (!sub) { html += openList(item.kind, item.n); sub = item.kind; }
      html += `<li>${inline(esc(item.text))}</li>`; i++; continue;
    }
    if (item) {
      flush(); closeSub(); if (liOpen) { html += '</li>'; liOpen = false; }
      if (top && top !== item.kind) { html += `</${top}>`; top = ''; }
      if (!top) { html += openList(item.kind, item.n); top = item.kind; }
      html += `<li>${inline(esc(item.text))}`; liOpen = true; i++; continue;
    }
    if (/^\s*$/.test(line)) {
      flush();
      if (top) {
        // Keep the list open across blank lines when it continues: another item, or an indented line.
        let j = i + 1; while (j < lines.length && /^\s*$/.test(lines[j])) j++;
        const next = j < lines.length ? lines[j] : '';
        if (next && (listItem(next) || /^\s{2,}\S/.test(next))) { i = j; continue; }
      }
      closeLists(); i++; continue;
    }
    if (top && liOpen && /^\s{2,}\S/.test(line)) {
      // An indented continuation line is more text for the open item, not a new paragraph.
      closeSub(); html += `<br>${inline(esc(line.trim()))}`; i++; continue;
    }
    if (top) closeLists();
    para.push(line); i++;
  }
  flush(); closeLists();
  return html;
}
