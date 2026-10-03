import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { LayerArtifact } from '../contracts';
import type { FileRecord, ModelCatalogue, TraceLine } from '../agentClient';
import { SUGGESTIONS } from '../agentBrain';
import type { AppTab } from '../uiVariant';
import { groupedSources, sourceHref, type SourceGroup } from '../answerFormat';

export interface ChatMessage {
  role: 'user' | 'agent';
  text?: string;
  html?: string;
  trace?: TraceLine[];
  artifacts?: FileRecord[];
  layers?: { id: string; label: string; source: string }[];
  response?: any;
  streaming?: boolean;
}

export type Mode = 'live' | 'local';
export interface AgentCfg { endpoint: string; uploadEndpoint: string; apiKey: string;
                            model?: string; provider?: string; reasoningEffort?: string;
                            codePeer?: string; codePeerModel?: string; orchestration?: string }

// The satellite-embedding models the service actually offers, grouped the way the choice
// matters: a precomputed model answers in seconds, an on-the-fly one runs the encoder. Six of
// the twenty, chosen to be worth demonstrating rather than exhaustive — "Which satellite
// embedding models can I use?" asks the agent for the full list, live.
const RS_MODELS: { group: string; ids: string[] }[] = [
  { group: 'Precomputed — answers in seconds', ids: ['gse', 'copernicus', 'tessera'] },
  { group: 'On the fly — runs the encoder', ids: ['clay', 'prithvi', 'terramind'] },
];

const RS_YEARS = ['2024', '2023', '2022', '2021', '2020', '2019', '2018'];

// embed_region truncates to _MAX_MODELS_PER_CALL SILENTLY (rs_embed_tools._MAX_MODELS_PER_CALL), so a
// sixth pick would vanish without a word. Stop at the cap in the UI, where it can be explained.
const RS_MAX_MODELS = 5;
// How tall the composer may grow before it starts scrolling. Four lines holds every staged
// operation question, and stops a pasted wall of text from eating the conversation above it.
const COMPOSER_MAX_ROWS = 4;
// The second group runs the encoder at request time. Every model in one embed_region call shares
// a SINGLE 600s budget (rs_embed_tools._TIMEOUT_S, 600s), so two of these together can blow it and lose the
// whole call — including the models that had already finished.
const RS_ONTHEFLY = ['clay', 'prithvi', 'terramind'];

// Written as the phrase that goes INTO the question, so the composed prompt reads like some-
// thing a person would type rather than a form serialised into a sentence. `over` is the same
// window across SEVERAL years, for the Change question.
const RS_SEASONS: { id: string; label: string; phrase: (y: string) => string; over: (ys: string) => string }[] = [
  { id: 'summer', label: 'Jun–Sep',    phrase: (y) => `June–September ${y}`,     over: (ys) => `June–September of ${ys}` },
  { id: 'spring', label: 'Mar–May',    phrase: (y) => `March–May ${y}`,          over: (ys) => `March–May of ${ys}` },
  { id: 'autumn', label: 'Sep–Nov',    phrase: (y) => `September–November ${y}`, over: (ys) => `September–November of ${ys}` },
  { id: 'year',   label: 'whole year', phrase: (y) => `the whole of ${y}`,       over: (ys) => `the whole of ${ys}` },
];

/** The years the Change question compares: `year` and two before it, two apart where there is
 *  room. Naively year-4/year-2 both clamp to the 2017 floor (GSE's first year), so 2018 and 2019
 *  each asked about "2017, 2017 and 2019" — and now that every year is put on the map, that is
 *  two identical layers, not just a clumsy sentence. Fill upward instead, and return however many
 *  distinct years actually exist: at 2018 there are only two. */
function rsChangeYears(year: string): string[] {
  const y = Number(year);
  const lo = 2017;
  const seen = new Set<number>();
  for (const v of [y - 4, y - 2, y]) seen.add(Math.max(lo, v));
  for (let v = lo; seen.size < 3 && v <= y; v++) seen.add(v);
  return [...seen].sort((a, b) => a - b).map(String);
}

/** "a", "a and b", "a, b and c" — the question has to read like English, not a serialised array. */
function listJoin(items: string[]): string {
  if (items.length < 2) return items[0] || '';
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

// The four operations, composed from the current model and period rather than frozen. The
// point of the demo is that these ARE parameters — a fixed "gse, June–September 2022" shows
// one cell of the space and hides that the rest exists.
function rsActions(models: string[], year: string, season: string) {
  const sn = RS_SEASONS.find((s) => s.id === season) || RS_SEASONS[0];
  const when = sn.phrase(year);
  const changeYears = rsChangeYears(year);
  // Only Embed takes several, and the reason is real rather than historical: embed_region sends
  // the whole list to /api/embed in ONE request and gets a layer back per model. The other three
  // are composed from ONE embedding — clustering it, differencing it across periods, running a
  // head on it — and each of those is defined against a single latent space, so a second model
  // would mean a second, separate analysis rather than a richer one. The rest read the first pick.
  // (There are no segment/change/predict tools to check against; see the composition contract in
  // embed_region's docstring.)
  const model = models[0] || 'gse';
  const many = models.length > 1;
  const modelList = listJoin(models);
  return [
    // Deliberately NOT "side by side" or "on a shared basis" for several models: that is the
    // documented trigger for align_embedding_colors, which REFUSES across models — each has its
    // own dimension and its own arbitrary frame (gse 64, tessera 128, terramind 384,
    // copernicus/prithvi 768), so a shared PCA basis is undefined, not merely unimplemented.
    // Each model gets its own layer and its own colours, and the sentence promises nothing more.
    { label: 'Embed',
      prompt: many
        // Every model renders the SAME footprint at opacity .85, so the layers stack and only the
        // top one shows — without being told, that reads as "only one model ran". Asking for the
        // list is what turns the stack into something navigable with the layer-list eye toggles.
        ? `Embed this drawn region with the ${modelList} models for ${when}, put each model's embedding on the map as its own layer, and list them — they cover the same ground, so only the top one is visible until I toggle the rest.`
        : `Embed this drawn region with the ${model} model for ${when} and put the embedding on the map.` },
    // Every operation below the first is COMPOSED from the embedding rather than naming a
    // one-shot tool, because those tools no longer exist: the agent embeds, then writes the
    // clustering / differencing / prediction against the package in code. The questions are
    // worded as outcomes, not as tool calls, so they keep working as the composition changes.
    { label: 'Segment',
      prompt: `Embed this drawn region with the ${model} model for ${when}, then cluster that embedding into 6 look-alike zones and put the zones on the map.` },
    // Composed rather than asking for a change tool, which returned a CSV and NO layer and threw
    // its per-year embeddings away — it told you THAT the place changed, not what changed.
    // Embedding each year instead puts them all on the map (start/end are part of the layer id,
    // so they do not collide), leaves reusable packages behind, and makes the change readable as
    // colour change. Here a shared basis IS meaningful: one model, one space, and
    // align_embedding_colors already numbers same-region repeats.
    { label: 'Change',
      prompt: `Embed this drawn region with the ${model} model for ${sn.over(listJoin(changeYears))}, put each year's embedding on the map on one shared colour basis, and work out from them how much the region changed year to year.` },
    { label: 'Predict',
      prompt: `Embed this drawn region with the ${model} model for ${when}, then run whatever pretrained heads cover ${model} on that embedding and report each prediction with its validation score. If no head covers ${model}, say so and tell me which models do.` },
  ];
}

// Starter questions for the remote-sensing tab. None of these needs a region drawn first — a
// starter that fails until you have done something else teaches the wrong thing about the tab.
const RS_SUGGESTIONS = [
  'Which satellite embedding models can I use?',
  'Embed Urbana, Illinois with the GSE model',
  'What can you do with satellite embeddings?',
  // Was "Compare Champaign and Urbana on a shared PCA basis" — the heaviest thing on the page
  // sitting where the lightest ought to be. It embedded two WHOLE cities (~30 km² each, tens of
  // thousands of tiles) before it could align anything, so the one starter a visitor is most
  // likely to click ran for minutes and spent imagery quota on the deployment's credential.
  // A 1 km box each keeps the point of it — one shared basis is what makes two regions
  // comparable by colour at all — at roughly a thirtieth of the ground, and names GSE, which
  // is precomputed and returns in seconds rather than being computed on the fly.
  'Compare 1 km boxes in Champaign and Urbana with GSE on a shared colour basis',
];

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  tab: AppTab;
  hasRegion: boolean;
  mapVisible: boolean;
  models: ModelCatalogue | null;
  layers: LayerArtifact[];
  mode: Mode;
  cfg: AgentCfg;
  spatial: boolean;
  showSettings: boolean;
  resolveUrl: (u: string) => string;
  onSend: (text: string) => void;
  onStop: () => void;
  onClearRegion: () => void;
  onToggleMap: () => void;
  onUpload: (files: File[]) => void;
  onSetMode: (m: Mode) => void;
  onSetCfg: (c: AgentCfg) => void;
  onSetSpatial: (v: boolean) => void;
}

const GROUP_LABEL: Record<SourceGroup, string> = {
  internal: 'I-GUIDE knowledge base', external: 'External open-data catalogs', web: 'Open web',
};
const isImg = (f: FileRecord) => f.kind === 'image' || /\.(png|jpe?g|gif|webp|bmp|avif)$/i.test(f.filename || f.download_url || '');

function Sources({ response }: { response: any }) {
  const groups = groupedSources(response);
  const order: SourceGroup[] = ['internal', 'external', 'web'];
  if (!order.some((k) => groups[k].length)) return null;
  return (
    <div className="srcs">
      <h4>Sources used</h4>
      {order.filter((k) => groups[k].length).map((k) => (
        // Collapsed by default: three groups expanded pushed the answer far up the panel, and
        // sources are a thing you consult, not a thing you read. <details> matches the Reasoning
        // block above and gives keyboard + screen-reader behaviour for free.
        <details key={k} className="grp">
          <summary className="hd">{GROUP_LABEL[k]}<span className="n">{groups[k].length} item{groups[k].length === 1 ? '' : 's'}</span><span className="chev">▾</span></summary>
          {groups[k].slice(0, 12).map((s, i) => {
            const title = String(s.title || s.doc_id || '(untitled)');
            // sourceHref, not s.url: internal knowledge elements carry no url of their own and
            // would otherwise render as plain text while external hits beside them are links.
            const url = sourceHref(s);
            const snip = String(s.abstract || s.snippet || s.contents || '').trim();
            return (
              <div key={i} className="it">
                <div className="t">{/^https?:\/\//i.test(url) ? <a href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</div>
                {snip && <div className="sn">{snip.length > 260 ? snip.slice(0, 260) + '…' : snip}</div>}
              </div>
            );
          })}
          {groups[k].length > 12 && <div className="sn">+{groups[k].length - 12} more not shown</div>}
        </details>
      ))}
    </div>
  );
}

/** One rendered row: a line, plus any lines folded underneath it. */
type Row = { line: TraceLine; folded?: TraceLine[] };

/** What the transcript SHOWS, folded from what it stored.
 *
 * Two ladders, both of which used to print in full. A ReAct round emits an "asking the model"
 * line before every tool call — eight identical rows naming the same model, between the rows
 * that said what actually happened; only the first survives, because the model is worth
 * stating once. And the graph narrates its own traversal: measured on a one-tool turn, nine of
 * fifteen rows were routing bookkeeping ("Routing the request", "Routed to orchestrate",
 * "Orchestrator agent started", "supervisor -> analyze (decision)", "Running analysis
 * workflow"), five to start and four to stop, against three rows that carried information.
 * A consecutive run of those collapses to its first line with the rest one click away.
 *
 * The first line, not the last, because a run opens by saying what is beginning — the last
 * line of the closing run is "Supervisor graph completed", which is the least useful string
 * in the set.
 *
 * Folded at RENDER, not at ingest. The array keeps every event, so the stored transcript stays
 * complete and a later reader is not looking at an edited record. */
export function foldTrace(trace: TraceLine[]): Row[] {
  let seenModelLine = false;
  const kept = trace.filter((t) => {
    if (t.kind !== 'llm') return true;
    if (seenModelLine) return false;
    seenModelLine = true;
    return true;
  });
  const rows: Row[] = [];
  for (const line of kept) {
    const prev = rows[rows.length - 1];
    if (line.kind === 'node' && prev && prev.line.kind === 'node') {
      (prev.folded ||= []).push(line);
    } else {
      rows.push({ line });
    }
  }
  return rows;
}

/** How many rows the transcript shows: one per fold row plus the rows folded under it. */
export function visibleSteps(trace: TraceLine[]): number {
  return foldTrace(trace).reduce((n, r) => n + 1 + (r.folded?.length || 0), 0);
}

// How much of a trace line shows before it is clamped. A traceback or a tool's argument dict
// runs to hundreds of characters, and a transcript where every row is a paragraph is unreadable
// — but the interesting half of a stack trace is the part that got cut. Clamped, clickable.
const TRACE_CLAMP = 140;

function TraceRow({ row }: { row: Row }) {
  const { line, folded } = row;
  const long = line.text.length > TRACE_CLAMP;
  const run = folded?.length || 0;
  // TWO independent states, because the two things a row can hide are not the same thing.
  // A run of steps starts SHOWN: this trace is read to find out what the agent did, and a
  // reader who has to click to see the steps is being asked to guess whether there is
  // anything behind the click. Collapsing is the deliberate act, not expanding.
  // A long message still starts CLAMPED — that one hides a 4000-char traceback or an argument
  // dict, and printing those in full is what made the transcript unreadable to begin with.
  const [openRun, setOpenRun] = useState(true);
  const [openText, setOpenText] = useState(false);
  const toggle = run > 0 ? () => setOpenRun((v) => !v) : () => setOpenText((v) => !v);
  const open = run > 0 ? openRun : openText;
  const expandable = long || run > 0;
  const label = run > 0
    ? (openRun ? 'Collapse these steps' : `Show ${run} more step${run === 1 ? '' : 's'}`)
    : (openText ? 'Collapse' : 'Show the whole message');
  return (
    <>
      <div
        className={`ln ${line.kind || ''}${expandable ? ' clampable' : ''}${open ? ' open' : ''}`}
        onClick={expandable ? toggle : undefined}
        role={expandable ? 'button' : undefined}
        tabIndex={expandable ? 0 : undefined}
        onKeyDown={expandable ? (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
        } : undefined}
        title={expandable ? label : undefined}
      >
        {openText || !long ? line.text : `${line.text.slice(0, TRACE_CLAMP)}…`}
        {run > 0 && <span className="more">{openRun ? `−${run}` : `+${run}`}</span>}
      </div>
      {openRun && folded?.map((f, i) => (
        <div className={`ln ${f.kind || ''} sub`} key={i}>{f.text}</div>
      ))}
    </>
  );
}

/** The name to show for a download — never the placeholder.
 *
 * `collectDownloads` fills `filename` with the literal string "download" when a tool result
 * carried a url and an id but no name, and several tools did exactly that — so a list of four
 * artifacts read "download · download · download · download". The tools now send their names;
 * this is the fallback for anything that still does not, and an id is at least unique and
 * traceable where a repeated word is neither. */
function displayName(f: FileRecord): string {
  const n = (f.filename || '').trim();
  if (n && n.toLowerCase() !== 'download') return n;
  return f.file_id ? `unnamed file (${f.file_id})` : 'unnamed file';
}

/** Extension-aware file glyph. Inline SVG rather than an emoji or an icon font: it inherits
 *  currentColor, so it stays legible in both themes without a second asset to load. */
function FileIcon({ name }: { name: string }) {
  const ext = (name.split('.').pop() || '').toLowerCase();
  const tag = ({ tif: 'TIF', tiff: 'TIF', png: 'PNG', jpg: 'JPG', jpeg: 'JPG', csv: 'CSV',
                 geojson: 'GEO', json: 'JSON', npz: 'NPZ', zip: 'ZIP', pdf: 'PDF',
                 txt: 'TXT', md: 'MD', html: 'HTM' } as Record<string, string>)[ext] || '';
  return (
    <svg className="dl-ico" viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"
         focusable="false">
      {/* A page with its corner turned. Two paths so the fold reads at 18px, where a single
          outline with a diagonal notch turns to mush. */}
      <path d="M14 2.5H7A1.5 1.5 0 0 0 5.5 4v16A1.5 1.5 0 0 0 7 21.5h10a1.5 1.5 0 0 0 1.5-1.5V7z"
            fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="M14 2.5V7h4.5" fill="none" stroke="currentColor" strokeWidth="1.4"
            strokeLinejoin="round" />
      {tag && <text x="12" y="17.4" textAnchor="middle" className="dl-ext">{tag}</text>}
    </svg>
  );
}

function AgentTurn({ m, resolveUrl }: { m: ChatMessage; resolveUrl: (u: string) => string }) {
  const imgs = (m.artifacts || []).filter(isImg).filter((f) => !(m.html || '').includes(f.file_id));
  const files = (m.artifacts || []).filter((f) => !isImg(f));
  const hasBody = m.html || m.text;
  return (
    <div className="turn">
      <div className="ai-label">I-GUIDE AI{m.streaming && <span className="spin" />}</div>
      {m.trace && m.trace.length > 0 && (
        <details className="reason" open={m.streaming}>
          {/* The tally counts what is SHOWN. Counting the stored array called a turn with seven
              tool calls "28 steps", most of them the folded model lines. */}
          {/* Counts what is SHOWN, and runs now show expanded — so this is every row again,
              minus the folded model ladder. Counting the collapsed rows instead called a
              fifteen-row transcript "7 steps" while fifteen rows sat under it. */}
          <summary>Reasoning<span className="tally">{m.streaming ? 'thinking…' : `${visibleSteps(m.trace)} steps`}</span><span className="chev">▾</span></summary>
          <div className="body">{foldTrace(m.trace).map((r, j) => <TraceRow key={j} row={r} />)}</div>
        </details>
      )}
      {(hasBody || imgs.length > 0 || m.response) && (
        <div className="answer-card">
          {m.html ? <div className="md" dangerouslySetInnerHTML={{ __html: m.html }} /> : m.text ? <div className="md"><p>{m.text}</p></div> : null}
          {m.streaming && !m.html && <span className="cursor">▋</span>}
          {imgs.length > 0 && (
            <div className="art-imgs">
              {imgs.map((f) => (
                <figure className="art" key={f.file_id}>
                  <a href={resolveUrl(f.download_url)} target="_blank" rel="noopener noreferrer"><img src={resolveUrl(f.download_url)} alt={f.filename} loading="lazy" /></a>
                  <figcaption><span className="nm">{f.filename}</span></figcaption>
                </figure>
              ))}
            </div>
          )}
          {files.length > 0 && (
            <div className="dl">
              <div className="dl-head">{files.length === 1 ? 'Download' : 'Downloads'}</div>
              {files.map((f) => (
                <a className="dl-item" key={f.file_id} href={resolveUrl(f.download_url)}
                   target="_blank" rel="noopener noreferrer" title={displayName(f)}>
                  <FileIcon name={displayName(f)} />
                  <span className="dl-name">{displayName(f)}</span>
                </a>
              ))}
            </div>
          )}
          {m.response && <Sources response={m.response} />}
        </div>
      )}
    </div>
  );
}

// How far from the bottom still counts as being AT the bottom. Never assume exactly 0:
// sub-pixel rounding and fractional device pixel ratios leave a residue of a pixel or two, and
// a reader a hair off the bottom still means "keep following".
const BOTTOM_SLACK_PX = 32;

export function ChatPanel(p: Props) {
  const [text, setText] = useState('');
  // Which operation is STAGED. Clicking one does not fire it: it writes the composed question
  // into the composer and opens the settings that built it, so the question can be read, tuned
  // and edited before it is asked. null means nothing is staged and the settings stay closed —
  // three selects above every conversation, for an operation nobody has chosen yet, is a form
  // where an offer belongs.
  const [rsOp, setRsOp] = useState<string | null>(null);
  // A LIST, because Embed takes several models in one call and returns a layer each. The other
  // three operations read the first entry — see rsActions.
  const [rsModels, setRsModels] = useState<string[]>(['gse']);
  const [rsYear, setRsYear] = useState('2022');
  const [rsSeason, setRsSeason] = useState('summer');

  // Stage an operation, or re-stage the current one after a setting changes.
  const stageRs = (label: string | null, models = rsModels, year = rsYear, season = rsSeason) => {
    if (!label) return;
    // Switching to an operation that takes ONE model drops the extra picks instead of leaving
    // a selection lit that the composed question will not use.
    const use = label === 'Embed' ? models : models.slice(0, 1);
    const action = rsActions(use, year, season).find((a) => a.label === label);
    if (!action) return;
    if (use.length !== models.length) setRsModels(use);
    setRsOp(label);
    setText(action.prompt);
  };

  // A setting changed: rewrite the staged question so the composer always shows what will
  // actually be sent. Only while something IS staged — otherwise changing a select would put
  // text into an empty composer the user never asked for.
  const setRsOption = (which: 'year' | 'season', value: string) => {
    const next = { year: rsYear, season: rsSeason, [which]: value } as
      { year: string; season: string };
    if (which === 'year') setRsYear(value);
    if (which === 'season') setRsSeason(value);
    stageRs(rsOp, rsModels, next.year, next.season);
  };

  // Only Embed composes a question about several models, so only Embed selects several. On any
  // other operation the chips behave as a radio group — the extra picks would be silently
  // dropped by a scalar-model tool, and a control that ignores what you told it is worse than
  // one that never offered.
  const rsMultiOk = rsOp === 'Embed';
  const toggleRsModel = (id: string) => {
    let next: string[];
    if (!rsMultiOk) next = [id];
    else if (rsModels.includes(id)) {
      // Never empty: a question needs a model, and the composer would read "with the  models".
      next = rsModels.length > 1 ? rsModels.filter((m) => m !== id) : rsModels;
    } else {
      if (rsModels.length >= RS_MAX_MODELS) return;   // the cap is enforced on the button too
      // Keep the declared order, so the sentence reads the same however they were clicked.
      const order = RS_MODELS.flatMap((g) => g.ids);
      next = order.filter((m) => m === id || rsModels.includes(m));
    }
    setRsModels(next);
    stageRs(rsOp, next, rsYear, rsSeason);
  };
  const scrollRef = useRef<HTMLDivElement>(null);
  // The composer grows with what is in it, to a ceiling of COMPOSER_MAX_ROWS lines. It was a
  // fixed 40px: a staged operation question runs to two or three lines, so the question you
  // were about to ask scrolled out of sight above the caret and could not be read before it was
  // sent — which is the entire point of staging it instead of firing it.
  const taRef = useRef<HTMLTextAreaElement>(null);
  // Whether the transcript is FOLLOWING new content. True while the reader is at the bottom,
  // false once they scroll up to read something. A ref, not state: it changes on every scroll
  // event and nothing renders from it, so re-rendering the transcript on each one would be
  // pure waste during a stream.
  const pinnedRef = useRef(true);
  // Until when a scroll may be treated as the READER's. Not every scroll event is one: the
  // browser re-anchors the scroll position by itself as streaming content reflows above the
  // viewport, and an earlier version of this took those for the reader scrolling away and
  // detached — permanently, because only a scroll back to the bottom re-attaches, and the
  // reader had never scrolled at all. So a scroll can only DETACH while the reader is
  // demonstrably driving; anything else can only ever re-attach.
  const drivingUntilRef = useRef(0);

  // Generous, to cover trackpad momentum after the last wheel event. Harmless if it is too
  // long: the only scrolls this window admits are ones that land away from the bottom, and a
  // programmatic follow always lands AT it.
  const markDriving = () => { drivingUntilRef.current = Date.now() + 1200; };

  const isAtBottom = () => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight <= BOTTOM_SLACK_PX;
  };

  const onScroll = () => {
    // Landing at the bottom always re-attaches, whoever caused it.
    if (isAtBottom()) { pinnedRef.current = true; return; }
    // Away from the bottom detaches only if the reader put it there.
    if (Date.now() < drivingUntilRef.current) pinnedRef.current = false;
  };

  useEffect(() => {
    const el = scrollRef.current;
    // Follow only when the reader is already at the bottom. This used to scroll
    // unconditionally, and with the full trace on it fires constantly — every reasoning step
    // patches `messages` with a new array — so reading anything above the fold was impossible:
    // the view was dragged back down mid-sentence every few hundred milliseconds. It fired
    // even with the reasoning block COLLAPSED, because the steps still change the array
    // whether or not anything visible grew.
    if (!el || !pinnedRef.current) return;
    // Instant, not smooth. A smooth scroll animates THROUGH positions that are not at the
    // bottom, and the handler below would read those as the reader moving away and unpin — so
    // the next step would silently stop following. Nothing looks smooth anyway when steps
    // arrive faster than the animation can finish.
    el.scrollTop = el.scrollHeight;
  }, [p.messages, p.busy]);

  // The pane's width is the user's now, and a narrower chat re-wraps every paragraph taller
  // while the browser holds scrollTop: a transcript that was following new content silently ends
  // up short of the bottom, and only a scroll back down re-attaches it. Re-assert the follow when
  // the WIDTH changes — height changes are the stream's business, handled above — and only when
  // it was already following, so a reader who scrolled up is never yanked down.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    let w = el.clientWidth;
    const ro = new ResizeObserver(() => {
      if (el.clientWidth === w) return;
      w = el.clientWidth;
      if (pinnedRef.current) el.scrollTop = el.scrollHeight;
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Runs before paint, so a staged question that needs three lines is never shown at one and
  // then jumped. Measured from the element's own computed style rather than from hardcoded
  // pixels, so changing the composer's font or padding cannot silently move the ceiling.
  useLayoutEffect(() => {
    const el = taRef.current;
    if (!el) return;
    const cs = getComputedStyle(el);
    const line = parseFloat(cs.lineHeight) || 22;
    const max = line * COMPOSER_MAX_ROWS
      + parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const wasScrolledTo = el.scrollTop;
    // Collapse first. scrollHeight on an element already tall enough reports the ELEMENT, not
    // the content, so without this the box would only ever grow and never shrink back — most
    // visibly after sending, when the text is gone but the height would remain.
    el.style.height = 'auto';
    const needed = el.scrollHeight;
    el.style.height = `${Math.min(needed, max)}px`;
    // Only scroll once the ceiling is actually reached; below it there is nothing to scroll and
    // a permanent scrollbar steals a few pixels from the text on some platforms.
    el.style.overflowY = needed > max ? 'auto' : 'hidden';
    if (needed > max) {
      // Measuring costs the scroll position: collapsing to `auto` removes the overflow, which
      // zeroes scrollTop, and past the ceiling that leaves the caret off-screen BELOW. That is
      // the same "cannot see what you are typing" the fixed height caused, just at the other
      // end. Typing at the end is the common case and wants the bottom; an edit in the middle
      // wants the view it already had.
      const atEnd = el.selectionStart === el.value.length
        && el.selectionEnd === el.value.length;
      el.scrollTop = atEnd ? el.scrollHeight : wasScrolledTo;
    }
  }, [text]);

  const send = (t: string) => {
    const v = t.trim();
    if (!v || p.busy) return;
    setText('');
    // The question has been asked, so nothing is staged any more — a later change to a select
    // must not rewrite a composer the user has moved on from.
    setRsOp(null);
    // Sending re-attaches. You asked the question; you want to watch the answer arrive, even
    // if you had scrolled up to re-read something before hitting enter.
    pinnedRef.current = true;
    p.onSend(v);
  };

  return (
    <section className="chat">
      {p.showSettings && (
        <div className="settings">
          <div className="grid">
            <label>Mode
              <select value={p.mode} onChange={(e) => p.onSetMode(e.target.value as Mode)}>
                <option value="live">Live agent (real backend)</option>
                <option value="local">Local demo (mock, offline)</option>
              </select>
            </label>
            <label>API key
              <input type="password" value={p.cfg.apiKey} placeholder="X-API-KEY (if required)" onChange={(e) => p.onSetCfg({ ...p.cfg, apiKey: e.target.value })} />
            </label>
            <label>Model
              <select value={p.cfg.model || ''}
                      onChange={(e) => {
                        const model = e.target.value;
                        // Carry the provider alongside the id: two providers could serve
                        // similarly-named models, and the server should not have to guess.
                        const owner = p.models?.providers.find(
                          (g: ModelCatalogue['providers'][number]) => g.models.includes(model));
                        // Repair the effort on switch. Leaving a stale value behind is how a
                        // pick of 'high' on one model turned every later turn into a 400 on a
                        // model that refuses any level once tools are attached — and the value
                        // persists to localStorage, so it outlived the reload too.
                        const legal = owner?.effort_options?.[model] || [];
                        const forced = owner?.effort_required?.[model];
                        const kept = forced
                          ? forced
                          : (p.cfg.reasoningEffort && legal.includes(p.cfg.reasoningEffort)
                              ? p.cfg.reasoningEffort : '');
                        p.onSetCfg({ ...p.cfg, model, provider: model ? (owner?.provider || '') : '',
                                     reasoningEffort: kept });
                      }}>
                <option value="">
                  Agent default{p.models ? ` (${p.models.default.model})` : ''}
                </option>
                {/* An unconfigured provider is shown DISABLED rather than dropped. Absent
                    reads as "this deployment cannot speak Claude", which is a different
                    thing from "nobody has put a key in yet" — and only one of those is
                    fixable from the .env. */}
                {(p.models?.providers || []).map((g: ModelCatalogue['providers'][number]) => (
                  <optgroup key={g.provider} disabled={!g.configured}
                    label={g.label
                      + (g.configured ? (g.caveat ? ` — ${g.caveat}` : '')
                                      : ` — needs ${g.needs || 'configuration'}`)
                      + (g.stale ? ' — list unavailable' : '')}>
                    {g.models.map((m: string) => (
                      <option key={m} value={m} disabled={!g.configured}>{m}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
            {/* The legal efforts depend on the model AND on tools being attached, which they
                always are here. Offer exactly what the API accepts: a model with one forced
                value is shown as fixed, and a model with no options shows no control. */}
            {(() => {
              const owner = p.models?.providers.find(
                (g: ModelCatalogue['providers'][number]) => g.models.includes(p.cfg.model || ''));
              const legal = owner?.effort_options?.[p.cfg.model || ''] || [];
              const forced = owner?.effort_required?.[p.cfg.model || ''];
              if (!legal.length) return null;
              if (forced) {
                return (
                  <label>Reasoning effort
                    <select value={forced} disabled title={`${p.cfg.model} requires reasoning_effort='${forced}' when tools are attached`}>
                      <option value={forced}>{forced} (required)</option>
                    </select>
                  </label>
                );
              }
              return (
                <label>Reasoning effort
                  <select value={p.cfg.reasoningEffort || ''}
                          onChange={(e) => p.onSetCfg({ ...p.cfg, reasoningEffort: e.target.value })}>
                    <option value="">Model default</option>
                    {legal.map((v: string) => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
              );
            })()}
            {/* Which SHAPE runs the turn: the supervisor routing between a search peer and
                an analyze peer, or one agent doing both in one context. Per-request so the
                two can be compared side by side rather than by restarting the deployment. */}
            <label>Orchestration
              <select value={p.cfg.orchestration || ''}
                      onChange={(e) => p.onSetCfg({ ...p.cfg, orchestration: e.target.value })}>
                <option value="">Server default</option>
                <option value="peers">Supervisor + peers</option>
                <option value="unified">Unified agent (experimental)</option>
              </select>
            </label>
            {/* A SECOND axis, deliberately its own control: `Model` picks what writes
                the answer, this picks what writes the code. A peer whose sandbox image or
                credential is missing is shown but disabled — absent from the list would
                read as "not built", which is a different problem from "not configured". */}
            {p.models?.code_peers && (
              <label>Code peer
                <select value={p.cfg.codePeer || ''}
                        onChange={(e) => {
                          // Drop a model chosen for the previous peer. Carrying 'opus'
                          // onto a backend that never heard of it is how a stale
                          // reasoning_effort used to 400 every later turn.
                          p.onSetCfg({ ...p.cfg, codePeer: e.target.value, codePeerModel: '' });
                        }}>
                  <option value="">
                    Server default ({p.models.code_peers.default})
                  </option>
                  {p.models.code_peers.peers.map((peer) => (
                    <option key={peer.id} value={peer.id} disabled={!peer.available}
                            title={peer.label}>
                      {/* Just the peer. Its model is the control next to this one, and
                          showing "claude (sonnet)" here read as a fixed pairing. */}
                      {peer.id}
                      {peer.available ? '' : ` — ${peer.reason || 'unavailable'}`}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {/* Only for a peer that HAS selectable models, and only once one is chosen:
                the built-in peer codes with whatever `Model` above already picked. */}
            {(() => {
              const peer = p.models?.code_peers?.peers.find((x) => x.id === p.cfg.codePeer);
              if (!peer?.models?.length) return null;
              return (
                <label>Peer model
                  <select value={p.cfg.codePeerModel || ''}
                          onChange={(e) => p.onSetCfg({ ...p.cfg, codePeerModel: e.target.value })}>
                    <option value="">Peer default{peer.model ? ` (${peer.model})` : ''}</option>
                    {peer.models.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </label>
              );
            })()}
            <label className="wide">Chat endpoint
              <input value={p.cfg.endpoint} onChange={(e) => p.onSetCfg({ ...p.cfg, endpoint: e.target.value })} />
            </label>
          </div>
          <label className="chk"><input type="checkbox" checked={p.spatial} onChange={(e) => p.onSetSpatial(e.target.checked)} /> Spatial tools (maps, OSM/Overpass, geo search) — off = pure chat</label>
        </div>
      )}

      {p.spatial && (
        <>
          {/* Only the control that DOES something. The hint that used to sit here repeated
              what the greeting and the numbered steps already say, and "◇ spatial on" reported
              a setting rather than offering an action — two lines of chrome above every
              conversation to say nothing new. An empty bar is not rendered at all. */}
          {p.hasRegion && (
            <div className="toolbar">
              <button onClick={p.onClearRegion}>Clear region</button>
            </div>
          )}
        </>
      )}

      <div className="transcript" ref={scrollRef}
        onScroll={onScroll}
        onWheel={markDriving} onTouchMove={markDriving}
        onMouseDown={markDriving} onKeyDown={markDriving}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); const fs = Array.from(e.dataTransfer.files || []); if (fs.length) p.onUpload(fs); }}>
        {p.messages.map((m, i) => m.role === 'user' ? (
          <div className="turn user" key={i}>
            <div className="who you">You</div>
            <div className="row right"><div className="bubble user">{m.text}</div></div>
          </div>
        ) : <AgentTurn key={i} m={m} resolveUrl={p.resolveUrl} />)}
        {p.busy && !p.messages.some((m) => m.streaming) && <div className="turn"><div className="ai-label">I-GUIDE AI<span className="spin" /></div></div>}
      </div>

      {p.messages.length <= 1 && (
        <div className="suggest">
          {(p.tab === 'rs' ? RS_SUGGESTIONS : SUGGESTIONS).map((s) => (
            <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}

      {/* The satellite-embedding operations belong to the RS-EMBED DEMO tab and appear only
          there. They sit directly above the composer, where the eye already is when you go to
          type; above the transcript they scrolled off the top of a long conversation and were
          never seen again. On that tab they are always on screen, disabled until a region
          exists, with the two steps spelled out: the tab exists to SHOW what can be done, and a
          hidden control demonstrates nothing.

          They used to appear on the Chat tab too, once a region had been drawn. Drawing a
          region is not the same as asking for satellite embeddings — you may have drawn it to
          ask any other question about the place — and the panel then sat between the
          conversation and the composer for the rest of the session. */}
      {p.spatial && p.tab === 'rs' && (
        <div className="rspanel demo">
          <ol className="rssteps">
            <li className={p.hasRegion ? 'done' : ''}>
              {p.mapVisible ? 'Right-click or right-drag on the map to draw a region'
                            : 'Open the map, then right-click or right-drag to draw a region'}
            </li>
            <li className={p.hasRegion ? '' : 'muted'}>Pick an operation</li>
          </ol>
          <div className="rsrow">
            {rsActions(rsModels, rsYear, rsSeason).map((a) => (
              <button key={a.label} className={`rsbtn ${rsOp === a.label ? 'on' : ''}`}
                disabled={p.busy || !p.hasRegion}
                aria-pressed={rsOp === a.label}
                title={p.hasRegion ? a.prompt : `Draw a region on the map first — then: ${a.prompt}`}
                onClick={() => stageRs(a.label)}>{a.label}</button>
            ))}
          </div>

          {/* The settings that BUILT the staged question, shown only once there is one. They
              sit below the operations and above the composer, between the choice they belong
              to and the text they rewrite. */}
          {rsOp && (
            <div className="rsopts">
              {/* Chips, not a <select>. The composer cannot be the readout of what is picked:
                  .box textarea is a fixed 40px with no auto-grow, so at a 380px pane it shows
                  about 22 characters and the model names sit well past that. The control has to
                  carry the selection itself, and a popover would hide it again. Its own full
                  width line, so adding models pushes nothing else around. */}
              <div className="rsmodels" role="group"
                   aria-label={rsMultiOk ? `Models — up to ${RS_MAX_MODELS}` : 'Model'}>
                {/* No visible label: it costs ~45px, which is exactly what pushes the six chips
                    from one line to three at the default 460px pane. The model ids are self-
                    identifying beside "year" and "months", each chip's title names its group, and
                    the group carries the accessible name. */}
                {RS_MODELS.flatMap((g) => g.ids).map((id) => {
                  const on = rsModels.includes(id);
                  const full = rsMultiOk && !on && rsModels.length >= RS_MAX_MODELS;
                  const group = RS_MODELS.find((g) => g.ids.includes(id))?.group ?? '';
                  return (
                    <button key={id} type="button"
                      className={`rschip ${on ? 'on' : ''}`}
                      // A radio group when the operation takes one model, a checkbox set when it
                      // takes several — so the semantics match what clicking actually does.
                      role={rsMultiOk ? 'checkbox' : 'radio'} aria-checked={on}
                      disabled={full}
                      title={full
                        ? `The service takes at most ${RS_MAX_MODELS} models in one call`
                        : `${id} — ${group}`}
                      onClick={() => toggleRsModel(id)}>{id}</button>
                  );
                })}
                {rsMultiOk && rsModels.filter((m) => RS_ONTHEFLY.includes(m)).length > 1 && (
                  <span className="rswarn" role="status">
                    two encoders share one 10-minute budget — if it runs out you lose the whole call
                  </span>
                )}
              </div>
              <label>year
                <select value={rsYear} onChange={(e) => setRsOption('year', e.target.value)}>
                  {RS_YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
                </select>
              </label>
              <label>months
                <select value={rsSeason} onChange={(e) => setRsOption('season', e.target.value)}>
                  {RS_SEASONS.map((sn) => <option key={sn.id} value={sn.id}>{sn.label}</option>)}
                </select>
              </label>
            </div>
          )}
        </div>
      )}

      <div className="composer">
        <div className="box">
          {/* Map lives with the composer controls, left of Attach: it is a thing you reach
              for while composing, and selecting a region needs the map open first. */}
          <button type="button" className={`circle map ${p.mapVisible ? 'on' : ''}`}
                  onClick={p.onToggleMap} aria-label={p.mapVisible ? 'Hide map' : 'Show map'}
                  title={p.mapVisible ? 'Hide the map'
                                 : 'Show the map, then right-click or right-drag on it to draw a region'}>
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                 strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 3.5 3.4 5.6a1 1 0 0 0-.65.94v13.2a.7.7 0 0 0 .95.65L9 18.5l6 2 5.6-2.1a1 1 0 0 0 .65-.94V4.26a.7.7 0 0 0-.95-.65L15 5.5Z" />
              <path d="M9 3.5v15M15 5.5v15" />
            </svg>
          </button>
          <label className="circle attach" title="Attach files">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.5l-8.5 8.5a5 5 0 01-7-7l9-9a3.5 3.5 0 015 5l-9 9a2 2 0 01-3-3l8-8" /></svg>
            <input type="file" multiple style={{ display: 'none' }} onChange={(e) => { const fs = Array.from(e.target.files || []); if (fs.length) p.onUpload(fs); (e.target as HTMLInputElement).value = ''; }} />
          </label>
          <textarea ref={taRef} value={text} rows={1}
            placeholder={p.mode === 'live' ? 'Ask me anything…' : 'Offline demo — try “show hospitals here”'}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(text); } }} />
          {p.busy ? (
            <button className="circle stop" onClick={p.onStop} title="Stop the agent" aria-label="Stop">
              <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
            </button>
          ) : (
            <button className="circle send" onClick={() => send(text)} disabled={!text.trim()} title="Send" aria-label="Send">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 11l5-5 5 5M12 6v12" /></svg>
            </button>
          )}
        </div>
        {/* Connection settings live on the top bar's gear alone now. Two entry points to one
            dialog, one of them under the composer where it competed with the notice, was one
            too many — and the notice is what belongs in the reading line under the box. */}
        <div className="footline">
          <span className="terms">I-GUIDE Platform Terms of Use apply. I-GUIDE AI Agent can make mistakes. Always double-check.</span>
        </div>
      </div>
    </section>
  );
}
