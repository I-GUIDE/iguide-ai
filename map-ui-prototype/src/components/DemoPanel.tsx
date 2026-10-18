import { useCallback, useEffect, useRef, useState } from 'react';

import type { AgentConfig } from '../agentClient';
import { DEMOS, demoBytes, demosByCategory, formatBytes, nextStep, type DemoCase } from '../demos';
import { type ExampleDraft, type ExampleListing, type UserExample, copyExample, createExample, deleteExample,
  forgetSharedExample, isUserExample, listExamples, openSharedExample, setPublished, shareExample, shareLink,
  unshareExample, updateExample } from '../userExamples';
import { ExampleEditor } from './ExampleEditor';

/** Whether the server knows who this is. 'none': the deployment identifies nobody (dev, demo). */
export type ExampleIdentity = 'unknown' | 'none' | 'signed-out' | 'signed-in';

type AnyCase = DemoCase | UserExample;

interface Props {
  busy: boolean;
  live: boolean;
  /** Where the agent is, for the user examples' own API. */
  cfg: AgentConfig;
  identity: ExampleIdentity;
  /** A share link's token from the page address, until it has been handled. */
  shareToken: string | null;
  onShareHandled: () => void;
  /** Start the case in a NEW conversation with its files attached. Resolves when they are. */
  onStart: (c: AnyCase) => Promise<void>;
  /** Ask a step's question, as if typed and sent. */
  onRun: (prompt: string) => void;
  /** Put a step's question in the composer to edit before sending. */
  onStage: (prompt: string) => void;
  /** Fold the sidebar to its rail. */
  onHide: () => void;
}

type Phase = 'idle' | 'loading' | 'ready' | 'error';
type Editor = null | { mode: 'new' } | { mode: 'edit'; id: string };

const EXPLAIN_FILES = 'Inspect and explain the files uploaded.';

const VISIBILITY_LABEL = { private: 'Private', link: 'Shared by link', published: 'Published' } as const;

function keyOf(c: AnyCase): string {
  return `${isUserExample(c) ? 'u' : 'b'}:${c.id}`;
}

function CaseCard({ c, onOpen, badge }: { c: AnyCase; onOpen: () => void; badge?: string }) {
  return (
    <button type="button" className="democard" onClick={onOpen} title={c.blurb}>
      <span className="demotitle">{c.title}{badge && <span className="exbadge">{badge}</span>}</span>
      {c.blurb && <span className="demoblurb">{c.blurb}</span>}
      <span className="demometa">
        {c.steps.length} step{c.steps.length > 1 ? 's' : ''} ·{' '}
        {c.files.length ? `${c.files.length} file${c.files.length > 1 ? 's' : ''}, ${formatBytes(demoBytes(c))}` : 'no upload'}
      </span>
    </button>
  );
}

// The Spatial Analysis tab: pick an example, start it (a fresh conversation with its files
// attached), then ask its steps in order. A step is a real question to the agent, sent exactly as
// written; "edit" puts it in the composer instead.
//
// Two kinds of example sit here. The built-in ones are a generated catalogue (demos.ts). The
// others are written by users and stored on the agent server (userExamples.ts): "My examples"
// and "Shared with me" are listed above the built-in ones, and an example an admin published
// is filed under its category beside them.
export function DemoPanel(p: Props) {
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState('');
  const [done, setDone] = useState<Set<number>>(new Set());
  const [listing, setListing] = useState<ExampleListing | null>(null);
  const [editor, setEditor] = useState<Editor>(null);
  const [notice, setNotice] = useState('');
  const [acting, setActing] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [copied, setCopied] = useState('');

  const userCases = listing ? [...listing.mine, ...listing.shared, ...listing.published] : [];
  const open: AnyCase | null = openKey
    ? (openKey.startsWith('u:') ? userCases.find((c) => `u:${c.id}` === openKey)
      : DEMOS.find((d) => `b:${d.id}` === openKey)) ?? null
    : null;

  const refresh = useCallback(async () => {
    if (!p.live || p.identity === 'unknown') return null;
    const l = await listExamples(p.cfg);
    if (l) setListing(l);
    return l;
  }, [p.live, p.identity, p.cfg.endpoint, p.cfg.apiKey]);
  useEffect(() => { void refresh(); }, [refresh]);

  const choose = (key: string | null) => {
    setOpenKey(key);
    setPhase('idle');
    setError('');
    setDone(new Set());
    setConfirmDelete(false);
    setCopied('');
  };

  // A share link opens the example it names, once the server has said who this is.
  const handledToken = useRef<string | null>(null);
  useEffect(() => {
    const token = p.shareToken;
    if (!token || !p.live || p.identity === 'unknown' || handledToken.current === token) return;
    if (p.identity === 'signed-out') {
      // Kept, not handled: once signed in, the same address opens it.
      setNotice('Someone shared an example with you. Sign in to open it.');
      return;
    }
    handledToken.current = token;
    if (p.identity === 'none') {
      setNotice('This deployment does not identify users, so a shared example cannot open here.');
      p.onShareHandled();
      return;
    }
    void (async () => {
      try {
        const ex = await openSharedExample(p.cfg, token);
        await refresh();
        setEditor(null);
        choose(`u:${ex.id}`);
        setNotice(ex.mine ? 'This is your own example.' : 'Added to "Shared with me".');
      } catch (e: any) {
        setNotice(`The shared link did not open an example: ${e?.message || e}`);
      } finally {
        p.onShareHandled();
      }
    })();
  }, [p.shareToken, p.live, p.identity]);

  const start = async (c: AnyCase) => {
    setPhase('loading');
    setError('');
    setDone(new Set());
    try {
      await p.onStart(c);
      setPhase('ready');
    } catch (e: any) {
      setPhase('error');
      setError(e?.message || String(e));
    }
  };

  const run = (k: number, prompt: string) => {
    p.onRun(prompt);
    setDone((s) => new Set(s).add(k));
  };

  /** Run one change to a user example; a failure is shown, never thrown. */
  const act = async (fn: () => Promise<void>) => {
    setActing(true);
    setNotice('');
    try { await fn(); } catch (e: any) { setNotice(e?.message || String(e)); } finally { setActing(false); }
  };

  const copyLink = async (token: string) => {
    const link = shareLink(token);
    let ok = false;
    try { await navigator.clipboard.writeText(link); ok = true; } catch { /* shown below to copy by hand */ }
    setCopied(link);
    setNotice(ok ? 'Link copied. Anyone signed in who opens it can run this example and copy it.'
              : 'Copy this link to share the example:');
  };

  const save = async (draft: ExampleDraft, files: File[]): Promise<string | null> => {
    try {
      const saved = editor?.mode === 'edit'
        ? await updateExample(p.cfg, editor.id, draft, files)
        : await createExample(p.cfg, draft, files);
      await refresh();
      setEditor(null);
      choose(`u:${saved.id}`);
      if (saved.unpublishedByEdit && !saved.published) setNotice('Saved. It is no longer published; an admin can publish it again.');
      return null;
    } catch (e: any) {
      return e?.message || String(e);
    }
  };

  if (editor) {
    const editing = editor.mode === 'edit' ? listing?.mine.find((c) => c.id === editor.id) : undefined;
    return <ExampleEditor editing={editing} limits={listing?.limits} onSave={save} onCancel={() => setEditor(null)} />;
  }

  if (!open) {
    const published = listing?.published ?? [];
    const groups = demosByCategory<AnyCase>([...DEMOS, ...published]);
    return (
      <div className="demopanel" aria-label="Spatial analysis examples">
        <div className="demohead">
          <strong className="demotitle">Spatial Analysis</strong>
          <button type="button" className="demoback fold" onClick={p.onHide}
                  title="Fold the examples to a rail so the conversation has the room">Hide</button>
        </div>
        <p className="demohint">
          Pick an example. Each one starts a new conversation with its data attached, then you ask
          its steps in order. {p.live ? '' : 'Connect to the live agent first: the offline demo cannot run these.'}
        </p>
        {notice && <p className="exnotice" role="status">{notice}</p>}

        {listing?.canCreate && (
          <section className="demogroup" aria-label="My examples">
            <div className="exgrouphead">
              <h3 className="demogrouphead">My examples</h3>
              <button type="button" className="demoback" onClick={() => { setNotice(''); setEditor({ mode: 'new' }); }}
                      disabled={listing.mine.length >= listing.limits.maxExamples}>+ New example</button>
            </div>
            {listing.mine.length ? (
              <div className="democards">
                {listing.mine.map((c) => (
                  <CaseCard key={c.id} c={c} onOpen={() => choose(keyOf(c))}
                            badge={c.visibility === 'private' ? undefined : VISIBILITY_LABEL[c.visibility]} />
                ))}
              </div>
            ) : (
              <p className="demohint">Write your own: its data, and the questions to ask in order. It is
                private until you share it.</p>
            )}
          </section>
        )}
        {p.identity === 'signed-out' && (
          <p className="demohint">Sign in to write your own examples and open ones shared with you.</p>
        )}

        {!!listing?.shared.length && (
          <section className="demogroup" aria-label="Shared with me">
            <h3 className="demogrouphead">Shared with me</h3>
            <div className="democards">
              {listing.shared.map((c) => <CaseCard key={c.id} c={c} onOpen={() => choose(keyOf(c))} />)}
            </div>
          </section>
        )}

        {groups.map((g) => (
          <section key={g.category} className="demogroup" aria-label={g.category}>
            <h3 className="demogrouphead">{g.category}</h3>
            <div className="democards">
              {g.cases.map((d) => (
                <CaseCard key={keyOf(d)} c={d} onOpen={() => choose(keyOf(d))}
                          badge={isUserExample(d) ? 'Community' : undefined} />
              ))}
            </div>
          </section>
        ))}
      </div>
    );
  }

  const next = nextStep(open, done);
  const user = isUserExample(open) ? open : null;
  const inShared = !!user && !!listing?.shared.some((c) => c.id === user.id);
  return (
    <div className="demopanel" aria-label={`Spatial analysis example: ${open.title}`}>
      <div className="demonav">
        <button type="button" className="demoback" onClick={() => { choose(null); setNotice(''); }}>← All examples</button>
        <button type="button" className="demoback fold" onClick={p.onHide}
                title="Fold the examples to a rail so the conversation has the room">Hide</button>
      </div>
      <p className="demoeyebrow">{open.category}{user && ` · ${user.mine ? VISIBILITY_LABEL[user.visibility] : user.published ? 'Community' : 'Shared with you'}`}</p>
      <h2 className="democasetitle">{open.title}</h2>
      {open.blurb && <p className="demoblurb democaseblurb">{open.blurb}</p>}

      {user && (
        <section className="exactions" aria-label="Example actions">
          {user.mine && (
            <>
              <button type="button" className="demoback" disabled={acting}
                      onClick={() => { setNotice(''); setEditor({ mode: 'edit', id: user.id }); }}>Edit</button>
              {user.shareToken ? (
                <>
                  <button type="button" className="demoback" disabled={acting}
                          onClick={() => void copyLink(user.shareToken!)}>Copy link</button>
                  <button type="button" className="demoback" disabled={acting}
                          title="Turn the link off: everyone it reached loses the example"
                          onClick={() => act(async () => {
                            await unshareExample(p.cfg, user.id);
                            await refresh();
                            setCopied('');
                            setNotice('Sharing stopped. The old link no longer opens this example.');
                          })}>Stop sharing</button>
                </>
              ) : (
                <button type="button" className="demoback" disabled={acting}
                        onClick={() => act(async () => {
                          const r = await shareExample(p.cfg, user.id);
                          await refresh();
                          await copyLink(r.shareToken);
                        })}>Share</button>
              )}
              {confirmDelete ? (
                <span className="exconfirm">Delete it for good?
                  <button type="button" className="demoback exdanger" disabled={acting}
                          onClick={() => act(async () => {
                            await deleteExample(p.cfg, user.id);
                            await refresh();
                            choose(null);
                            setNotice(`Deleted "${user.title}".`);
                          })}>Delete</button>
                  <button type="button" className="demoback" onClick={() => setConfirmDelete(false)}>Keep</button>
                </span>
              ) : (
                <button type="button" className="demoback" disabled={acting} onClick={() => setConfirmDelete(true)}>Delete</button>
              )}
            </>
          )}
          {!user.mine && listing?.canCreate && (
            <button type="button" className="demoback" disabled={acting}
                    onClick={() => act(async () => {
                      const c = await copyExample(p.cfg, user.id);
                      await refresh();
                      choose(`u:${c.id}`);
                      setNotice('Copied into "My examples". The copy is yours to edit.');
                    })}>Copy to my examples</button>
          )}
          {inShared && (
            <button type="button" className="demoback" disabled={acting}
                    onClick={() => act(async () => {
                      await forgetSharedExample(p.cfg, user.id);
                      await refresh();
                      choose(null);
                      setNotice(`Removed "${user.title}" from your list.`);
                    })}>Remove from my list</button>
          )}
          {(listing?.canPublish || (user.mine && user.published)) && (
            <button type="button" className="demoback" disabled={acting}
                    title={user.published ? 'Take it off everyone\'s list' : 'List it for everyone, beside the built-in examples'}
                    onClick={() => act(async () => {
                      await setPublished(p.cfg, user.id, !user.published);
                      await refresh();
                      setNotice(user.published ? 'Unpublished.' : 'Published: everyone now sees it under its category.');
                    })}>{user.published ? 'Unpublish' : 'Publish'}</button>
          )}
        </section>
      )}
      {notice && <p className="exnotice" role="status">{notice}</p>}
      {copied && <input className="exlink" readOnly value={copied} aria-label="Share link"
                        onFocus={(e) => e.currentTarget.select()} />}

      {open.files.length > 0 && (
        <section className="demosection" aria-label="Example data">
          <div className="demosectionhead">
            <h3>Data</h3>
            {/* Once the files are in the conversation, the agent can be asked what they are
                before any step: the steps no longer describe the data themselves. */}
            {phase === 'ready' && (
              <button type="button" className="demoexplain" disabled={p.busy}
                      title={EXPLAIN_FILES} onClick={() => p.onRun(EXPLAIN_FILES)}>Explain files</button>
            )}
          </div>
          <ul className="demofilelist">
            {open.files.map((f) => (
              <li key={f.name}><span className="demofname" title={f.name}>{f.name}</span>
                <span className="demofsize">{formatBytes(f.bytes)}</span></li>
            ))}
          </ul>
        </section>
      )}

      {phase !== 'ready' && (
        <div className="demostart">
          <button type="button" className="rsbtn on" disabled={p.busy || phase === 'loading' || !p.live}
                  onClick={() => start(open)}>
            {phase === 'loading' ? 'Starting…' : open.files.length ? 'Upload the example data and start a new conversation' : 'Start a new conversation'}
          </button>
          {phase === 'error' && <span className="rswarn" role="alert">Could not start: {error}</span>}
        </div>
      )}

      <section className="demosection" aria-label="Steps">
        <div className="demosectionhead"><h3>Steps</h3></div>
        <ol className="demosteps">
          {open.steps.map((s, k) => {
            const state = done.has(k) ? 'done' : k === next ? 'next' : '';
            const locked = phase !== 'ready' || p.busy;
            // Only the next step spells out its question; the rest show it on hover, so a
            // four-step case fits without pushing the conversation off the screen.
            const showPrompt = k === next;
            return (
              <li key={k} className={state}>
                <div className="demosteprow">
                  <span className="demosteplabel" title={s.prompt}>{s.label}</span>
                  <button type="button" className="demoedit" disabled={locked}
                          aria-label={`Edit step ${k + 1} before asking`}
                          onClick={() => p.onStage(s.prompt)}>edit</button>
                  <button type="button" className="demosend" disabled={locked}
                          aria-label={`${done.has(k) ? 'Send again' : 'Send'} step ${k + 1}: ${s.label}`}
                          title={phase !== 'ready' ? 'Start the example first' : s.prompt}
                          onClick={() => run(k, s.prompt)}>{done.has(k) ? 'Send again' : 'Send'}</button>
                </div>
                {showPrompt && <p className="demoprompt">{s.prompt}</p>}
              </li>
            );
          })}
        </ol>
      </section>

      {open.tags.length > 0 && (
        <section className="demosection" aria-label="Topics">
          <div className="demosectionhead"><h3>Topics</h3></div>
          <div className="demotags">
            {open.tags.map((t) => <span key={t} className="demotag">{t}</span>)}
            {/* `reliability` ("N of N harness runs") stays in the catalogue for the demo check,
                but is not shown: it is a test statistic, not something a visitor needs. */}
          </div>
        </section>
      )}
    </div>
  );
}
