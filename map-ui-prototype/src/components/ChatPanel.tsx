import { useEffect, useRef, useState } from 'react';
import type { LayerArtifact } from '../contracts';
import type { FileRecord, ModelCatalogue, TraceLine } from '../agentClient';
import { SUGGESTIONS } from '../agentBrain';
import { groupedSources, type SourceGroup } from '../answerFormat';

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
                            model?: string; provider?: string; reasoningEffort?: string }

const RS_ACTIONS = [
  { label: 'Embed',   prompt: 'Embed this drawn region with the gse model for June–September 2022 and put the embedding on the map.' },
  { label: 'Segment', prompt: 'Segment this drawn region into 6 look-alike zones from its satellite embedding and show it on the map.' },
  { label: 'Change',  prompt: 'How much did this drawn region change across 2018, 2020, 2022 and 2024 according to its satellite embeddings?' },
  { label: 'Predict', prompt: 'Run the available pretrained heads on this drawn region and report the predictions with their validation scores.' },
];

interface Props {
  messages: ChatMessage[];
  busy: boolean;
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
  onToggleSettings: () => void;
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
        <div key={k} className="grp">
          <div className="hd">{GROUP_LABEL[k]}<span className="n">{groups[k].length} item{groups[k].length === 1 ? '' : 's'}</span></div>
          {groups[k].slice(0, 12).map((s, i) => {
            const title = String(s.title || s.doc_id || '(untitled)');
            const url = String(s.url || '');
            const snip = String(s.abstract || s.snippet || s.contents || '').trim();
            return (
              <div key={i} className="it">
                <div className="t">{/^https?:\/\//i.test(url) ? <a href={url} target="_blank" rel="noopener noreferrer">{title}</a> : title}</div>
                {snip && <div className="sn">{snip.length > 260 ? snip.slice(0, 260) + '…' : snip}</div>}
              </div>
            );
          })}
          {groups[k].length > 12 && <div className="sn">+{groups[k].length - 12} more not shown</div>}
        </div>
      ))}
    </div>
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
          <summary>Reasoning<span className="tally">{m.streaming ? 'thinking…' : `${m.trace.length} steps`}</span><span className="chev">▾</span></summary>
          <div className="body">{m.trace.map((t, j) => <div key={j} className={`ln ${t.kind || ''}`}>{t.text}</div>)}</div>
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
          {files.length > 0 && <div className="files">{files.map((f) => <a key={f.file_id} href={resolveUrl(f.download_url)} target="_blank" rel="noopener noreferrer">{f.filename}</a>)}</div>}
          {m.response && <Sources response={m.response} />}
        </div>
      )}
    </div>
  );
}

export function ChatPanel(p: Props) {
  const [text, setText] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [p.messages, p.busy]);

  const send = (t: string) => { const v = t.trim(); if (!v || p.busy) return; setText(''); p.onSend(v); };

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
                {(p.models?.providers || [])
                  .filter((g: ModelCatalogue['providers'][number]) => g.configured)
                  .map((g: ModelCatalogue['providers'][number]) => (
                  <optgroup key={g.provider} label={g.label + (g.stale ? ' — list unavailable' : '')}>
                    {g.models.map((m: string) => <option key={m} value={m}>{m}</option>)}
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
            <label className="wide">Chat endpoint
              <input value={p.cfg.endpoint} onChange={(e) => p.onSetCfg({ ...p.cfg, endpoint: e.target.value })} />
            </label>
          </div>
          <label className="chk"><input type="checkbox" checked={p.spatial} onChange={(e) => p.onSetSpatial(e.target.checked)} /> Spatial tools (maps, OSM/Overpass, geo search) — off = pure chat</label>
        </div>
      )}

      {p.spatial && (
        <>
          <div className="toolbar">
            {p.mapVisible && <span className="hint">right-drag the map to select a region</span>}
            {/* Shown only when there IS a region: a permanently greyed-out button is just
                clutter — the toolbar should offer what can actually be done right now. */}
            {p.hasRegion && <button onClick={p.onClearRegion}>Clear region</button>}
            <span className={p.hasRegion ? 'rstat on' : 'rstat'}>{p.hasRegion ? '● region set' : '◇ spatial on'}</span>
          </div>
          {p.hasRegion && (
            <div className="rsrow" title="Run on the drawn region">
              <span className="rslabel">🛰 satellite embedding</span>
              {RS_ACTIONS.map((a) => (
                <button key={a.label} className="rsbtn" disabled={p.busy}
                  onClick={() => p.onSend(a.prompt)}>{a.label}</button>
              ))}
            </div>
          )}
        </>
      )}

      <div className="transcript" ref={scrollRef}
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
        <div className="suggest">{SUGGESTIONS.map((s) => <button key={s} className="chip" onClick={() => send(s)}>{s}</button>)}</div>
      )}

      <div className="composer">
        <div className="box">
          {/* Map lives with the composer controls, left of Attach: it is a thing you reach
              for while composing, and selecting a region needs the map open first. */}
          <button type="button" className={`circle map ${p.mapVisible ? 'on' : ''}`}
                  onClick={p.onToggleMap} aria-label={p.mapVisible ? 'Hide map' : 'Show map'}
                  title={p.mapVisible ? 'Hide the map' : 'Show the map, then right-drag on it to select a region'}>
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
          <textarea value={text}
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
        <div className="footline">
          <button className="conn" onClick={p.onToggleSettings}>⚙ Connection</button>
          <span className="terms">I-GUIDE Platform Terms of Use apply. Smart Search can make mistakes. Always double-check.</span>
        </div>
      </div>
    </section>
  );
}
