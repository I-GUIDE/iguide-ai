import { useState } from 'react';

import { DEMO_CATEGORIES, formatBytes } from '../demos';
import { type ExampleDraft, type ExampleLimits, type UserExample, draftFrom, draftProblem, emptyDraft,
  moveStep } from '../userExamples';

interface Props {
  /** The example being edited; absent for a new one. */
  editing?: UserExample;
  limits?: ExampleLimits;
  /** Save the draft. Resolves to an error message, or null when saved. */
  onSave: (draft: ExampleDraft, newFiles: File[]) => Promise<string | null>;
  onCancel: () => void;
}

// The "New example" / "Edit example" form: what the example is, its data, and its steps in order.
// A step is a question sent to the agent exactly as written, so the label is only what the list
// shows; the question is what runs.
export function ExampleEditor(p: Props) {
  const [draft, setDraft] = useState<ExampleDraft>(() => (p.editing ? draftFrom(p.editing) : emptyDraft(DEMO_CATEGORIES[0])));
  const [newFiles, setNewFiles] = useState<File[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const set = (patch: Partial<ExampleDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const setStep = (k: number, patch: Partial<{ label: string; prompt: string }>) =>
    set({ steps: draft.steps.map((s, i) => (i === k ? { ...s, ...patch } : s)) });

  const kept = p.editing?.files.filter((f) => draft.keepFiles.includes(f.name)) ?? [];
  const totalBytes = kept.reduce((s, f) => s + f.bytes, 0) + newFiles.reduce((s, f) => s + f.size, 0);
  // The category list offers the built-in ones; an example already filed under another keeps it.
  const categories: string[] = [...DEMO_CATEGORIES];
  if (draft.category && !categories.includes(draft.category)) categories.push(draft.category);

  const addFiles = (list: FileList | null) => {
    if (!list) return;
    const added = Array.from(list);
    // A file with the name of one already chosen or kept replaces it, as the server does.
    setNewFiles((cur) => [...cur.filter((f) => !added.some((a) => a.name === f.name)), ...added]);
    set({ keepFiles: draft.keepFiles.filter((n) => !added.some((a) => a.name === n)) });
  };

  const save = async () => {
    const problem = draftProblem(draft, newFiles, p.limits);
    if (problem) { setError(problem); return; }
    setSaving(true);
    setError('');
    const err = await p.onSave(draft, newFiles);
    setSaving(false);
    if (err) setError(err);
  };

  return (
    <form className="demopanel exampleform" aria-label={p.editing ? 'Edit example' : 'New example'}
          onSubmit={(e) => { e.preventDefault(); void save(); }}>
      <div className="demonav">
        <button type="button" className="demoback" onClick={p.onCancel}>← Cancel</button>
      </div>
      <h2 className="democasetitle">{p.editing ? 'Edit example' : 'New example'}</h2>
      {p.editing?.published && (
        <p className="demohint">This example is published. Saving a change takes it back to unpublished
          until an admin publishes it again, unless you are one.</p>
      )}

      <label className="exfield">
        <span>Title</span>
        <input value={draft.title} maxLength={120} required onChange={(e) => set({ title: e.target.value })}
               placeholder="e.g. Flood exposure of schools in Houston" />
      </label>
      <label className="exfield">
        <span>Description</span>
        <textarea value={draft.description} maxLength={2000} rows={3}
                  onChange={(e) => set({ description: e.target.value })}
                  placeholder="What the example shows and what data it uses" />
      </label>
      <label className="exfield">
        <span>Category</span>
        <select value={draft.category} onChange={(e) => set({ category: e.target.value })}>
          {categories.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>
      <label className="exfield">
        <span>Topics <small>(comma-separated)</small></span>
        <input value={draft.tags} onChange={(e) => set({ tags: e.target.value })}
               placeholder="buffer, overlay, census" />
      </label>

      <section className="demosection" aria-label="Example data">
        <div className="demosectionhead"><h3>Data</h3>
          <span className="demofsize">{formatBytes(totalBytes)}{p.limits ? ` of ${formatBytes(p.limits.maxBytes)}` : ''}</span>
        </div>
        <p className="demohint">Optional. Uploaded into each new conversation the example starts. Leave it
          empty when the agent should fetch its own data.</p>
        <ul className="demofilelist">
          {kept.map((f) => (
            <li key={`k-${f.name}`}><span className="demofname" title={f.name}>{f.name}</span>
              <span className="demofsize">{formatBytes(f.bytes)}</span>
              <button type="button" className="exicon" aria-label={`Remove ${f.name}`}
                      onClick={() => set({ keepFiles: draft.keepFiles.filter((n) => n !== f.name) })}>×</button></li>
          ))}
          {newFiles.map((f) => (
            <li key={`n-${f.name}`}><span className="demofname" title={f.name}>{f.name}</span>
              <span className="demofsize">{formatBytes(f.size)} · new</span>
              <button type="button" className="exicon" aria-label={`Remove ${f.name}`}
                      onClick={() => setNewFiles((cur) => cur.filter((x) => x !== f))}>×</button></li>
          ))}
        </ul>
        <label className="exaddfile">
          <input type="file" multiple aria-label="Add data files"
                 onChange={(e) => { addFiles(e.target.files); e.target.value = ''; }} />
          <span>+ Add files</span>
        </label>
      </section>

      <section className="demosection" aria-label="Steps">
        <div className="demosectionhead"><h3>Steps</h3></div>
        <ol className="exsteps">
          {draft.steps.map((s, k) => (
            <li key={k}>
              <div className="exsteprow">
                <span className="exstepn">{k + 1}.</span>
                <input value={s.label} maxLength={120} aria-label={`Step ${k + 1} label`} placeholder="Short label"
                       onChange={(e) => setStep(k, { label: e.target.value })} />
                <button type="button" className="exicon" disabled={k === 0} aria-label={`Move step ${k + 1} up`}
                        onClick={() => set({ steps: moveStep(draft.steps, k, -1) })}>↑</button>
                <button type="button" className="exicon" disabled={k === draft.steps.length - 1}
                        aria-label={`Move step ${k + 1} down`}
                        onClick={() => set({ steps: moveStep(draft.steps, k, 1) })}>↓</button>
                <button type="button" className="exicon" disabled={draft.steps.length === 1}
                        aria-label={`Delete step ${k + 1}`}
                        onClick={() => set({ steps: draft.steps.filter((_, i) => i !== k) })}>×</button>
              </div>
              <textarea value={s.prompt} rows={3} maxLength={4000} aria-label={`Step ${k + 1} question`}
                        placeholder="The question to ask the agent, exactly as it will be sent"
                        onChange={(e) => setStep(k, { prompt: e.target.value })} />
            </li>
          ))}
        </ol>
        <button type="button" className="demoback" disabled={!!p.limits && draft.steps.length >= p.limits.maxSteps}
                onClick={() => set({ steps: [...draft.steps, { label: '', prompt: '' }] })}>+ Add step</button>
      </section>

      <div className="demostart">
        <button type="submit" className="rsbtn on" disabled={saving}>
          {saving ? 'Saving…' : p.editing ? 'Save changes' : 'Create example'}</button>
        {error && <span className="rswarn" role="alert">{error}</span>}
      </div>
    </form>
  );
}
