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
  let html = ''; let i = 0; let inUl = false; let inOl = false; let para: string[] = [];
  const closeLists = () => { if (inUl) { html += '</ul>'; inUl = false; } if (inOl) { html += '</ol>'; inOl = false; } };
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
    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ul) { flush(); if (inOl) { html += '</ol>'; inOl = false; } if (!inUl) { html += '<ul>'; inUl = true; } html += `<li>${inline(esc(ul[1]))}</li>`; i++; continue; }
    const ol = line.match(/^\s*\d+\.\s+(.*)$/);
    if (ol) { flush(); if (inUl) { html += '</ul>'; inUl = false; } if (!inOl) { html += '<ol>'; inOl = true; } html += `<li>${inline(esc(ol[1]))}</li>`; i++; continue; }
    if (/^\s*$/.test(line)) { flush(); closeLists(); i++; continue; }
    para.push(line); i++;
  }
  flush(); closeLists();
  return html;
}
