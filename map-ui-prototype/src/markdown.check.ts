/** A runnable check of how answers' lists render. `npm run check:markdown`.
 *
 * No test runner here, and the failure is quiet: a broken list still renders, just numbered wrong.
 * On a prod answer on 2026-10-13 every numbered item read "1.", because a blank line between items,
 * or an indented sub-bullet under one, closed the <ol> after each item.
 */
import { renderMarkdown } from './markdown';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};
const md = (s: string) => renderMarkdown(s, (u) => u);
const count = (h: string, tag: string) => (h.match(new RegExp(`<${tag}[ >]`, 'g')) || []).length;

eq('a tight list is one list',
   md('1. a\n2. b\n3. c'), '<ol><li>a</li><li>b</li><li>c</li></ol>');
eq('a loose list (blank lines between items) is still one list',
   md('1. a\n\n2. b\n\n3. c'), '<ol><li>a</li><li>b</li><li>c</li></ol>');
eq('sub-bullets nest inside their item',
   md('1. **A**\n   - x\n2. **B**\n   - y'),
   '<ol><li><strong>A</strong><ul><li>x</li></ul></li><li><strong>B</strong><ul><li>y</li></ul></li></ol>');
eq('a loose list with sub-bullets is one list of three',
   count(md('1. A\n   - x\n\n2. B\n   - y\n\n3. C'), 'ol'), 1);
eq('numbering that starts later keeps its start',
   md('4. d\n5. e'), '<ol start="4"><li>d</li><li>e</li></ol>');
eq('an indented continuation line stays in its item',
   md('1. first\n   more text\n2. second'), '<ol><li>first<br>more text</li><li>second</li></ol>');
eq('a paragraph after a list closes it',
   md('1. a\n2. b\n\nAfter.'), '<ol><li>a</li><li>b</li></ol><p>After.</p>');
eq('a heading between two lists starts a new one',
   count(md('1. a\n\n## H\n\n1. b'), 'ol'), 2);
eq('bullets then numbers are two lists',
   md('- a\n1. b'), '<ul><li>a</li></ul><ol><li>b</li></ol>');
eq('"1)" style numbering is a list too',
   md('1) a\n2) b'), '<ol><li>a</li><li>b</li></ol>');
eq('plain paragraphs are untouched',
   md('Hello\nworld'), '<p>Hello<br>world</p>');

if (bad) { console.error(`\n${bad} check(s) failed`); process.exit(1); }
console.log('\nall markdown checks passed');
