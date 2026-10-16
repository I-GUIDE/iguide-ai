import { useState } from 'react';

import { DEMOS, demoBytes, demosByCategory, formatBytes, nextStep, type DemoCase } from '../demos';

interface Props {
  busy: boolean;
  live: boolean;
  /** Start the case in a NEW conversation with its files attached. Resolves when they are. */
  onStart: (c: DemoCase) => Promise<void>;
  /** Ask a step's question, as if typed and sent. */
  onRun: (prompt: string) => void;
  /** Put a step's question in the composer to edit before sending. */
  onStage: (prompt: string) => void;
  /** Fold the sidebar to its rail. */
  onHide: () => void;
}

type Phase = 'idle' | 'loading' | 'ready' | 'error';

const EXPLAIN_FILES = 'Inspect and explain the files uploaded.';

// The Demos tab: pick a case, start it (a fresh conversation with the case's files attached),
// then ask its steps in order. A step is a real question to the agent, sent exactly as written;
// "edit" puts it in the composer instead. It sits above the composer like the rs-embed panel.
export function DemoPanel(p: Props) {
  const [openId, setOpenId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState('');
  const [done, setDone] = useState<Set<number>>(new Set());
  const open = DEMOS.find((d) => d.id === openId) ?? null;

  const choose = (id: string | null) => {
    setOpenId(id);
    setPhase('idle');
    setError('');
    setDone(new Set());
  };

  const start = async (c: DemoCase) => {
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

  if (!open) {
    return (
      <div className="demopanel" aria-label="Demo cases">
        <div className="demohead">
          <strong className="demotitle">Demos</strong>
          <button type="button" className="demoback fold" onClick={p.onHide}
                  title="Fold the demos to a rail so the conversation has the room">Hide</button>
        </div>
        <p className="demohint">
          Pick a case. Each one starts a new conversation with its data attached, then you ask
          its steps in order. {p.live ? '' : 'Connect to the live agent first: the offline demo cannot run these.'}
        </p>
        {demosByCategory().map((g) => (
          <section key={g.category} className="demogroup" aria-label={g.category}>
            <h3 className="demogrouphead">{g.category}</h3>
            <div className="democards">
              {g.cases.map((d) => (
                <button key={d.id} type="button" className="democard" onClick={() => choose(d.id)}
                        title={d.blurb}>
                  <span className="demotitle">{d.title}</span>
                  <span className="demoblurb">{d.blurb}</span>
                  <span className="demometa">
                    {d.steps.length} step{d.steps.length > 1 ? 's' : ''} ·{' '}
                    {d.files.length ? `${d.files.length} file${d.files.length > 1 ? 's' : ''}, ${formatBytes(demoBytes(d))}` : 'no upload'}
                  </span>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    );
  }

  const next = nextStep(open, done);
  return (
    <div className="demopanel" aria-label={`Demo: ${open.title}`}>
      <div className="demohead">
        <button type="button" className="demoback" onClick={() => choose(null)}>← All demos</button>
        <strong className="demotitle">{open.title}</strong>
        <button type="button" className="demoback fold" onClick={p.onHide}
                title="Fold the demos to a rail so the conversation has the room">Hide</button>
      </div>
      <p className="demoblurb">{open.blurb}</p>
      {open.files.length > 0 && (
        <div className="demofilesrow">
          <p className="demofiles">
            Data: {open.files.map((f) => `${f.name} (${formatBytes(f.bytes)})`).join(', ')}
          </p>
          {/* Once the files are in the conversation, the agent can be asked what they are
              before any step: the steps no longer describe the data themselves. */}
          {phase === 'ready' && (
            <button type="button" className="rsbtn demoexplain" disabled={p.busy}
                    title={EXPLAIN_FILES} onClick={() => p.onRun(EXPLAIN_FILES)}>Explain files</button>
          )}
        </div>
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
      <ol className="demosteps">
        {open.steps.map((s, k) => {
          const state = done.has(k) ? 'done' : k === next ? 'next' : '';
          const locked = phase !== 'ready' || p.busy;
          // Only the next step spells out its question; the rest show it on hover, so a
          // three-step case fits without pushing the conversation off the screen.
          const showPrompt = k === next;
          return (
            <li key={k} className={state}>
              <div className="demosteprow">
                <span className="demosteplabel">{s.label}</span>
                <button type="button" className="rsbtn" disabled={locked}
                        aria-label={`${done.has(k) ? 'Send again' : 'Send'} step ${k + 1}: ${s.label}`}
                        title={phase !== 'ready' ? 'Start the demo first' : s.prompt}
                        onClick={() => run(k, s.prompt)}>{done.has(k) ? 'Send again' : 'Send'}</button>
                <button type="button" className="demoedit" disabled={locked}
                        aria-label={`Edit step ${k + 1} before asking`}
                        onClick={() => p.onStage(s.prompt)}>edit</button>
              </div>
              {showPrompt && <div className="demoprompt">{s.prompt}</div>}
            </li>
          );
        })}
      </ol>
      <div className="demotags">
        {open.tags.map((t) => <span key={t} className="demotag">{t}</span>)}
        {/* `reliability` ("N of N harness runs") stays in the catalogue for the demo check,
            but is not shown: it is a test statistic, not something a visitor needs. */}
      </div>
    </div>
  );
}
