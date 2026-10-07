/** A runnable check for what happens to a saved model the catalogue stops offering.
 *  `npm run check:models`.
 *
 * Same reasoning as check:auth. A <select> whose value matches no option shows "Agent default"
 * while every turn still sends the saved id, so a screenshot of a wrong answer here looks right.
 */
import type { ModelCatalogue } from './agentClient';
import { hiddenNote, unavailableChoice } from './modelChoice';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

const REFUSED = 'AnvilGPT refuses tool calls for it (its server runs without a tool-call parser)';
const cat = (anvil: Partial<ModelCatalogue['providers'][number]>): ModelCatalogue => ({
  default: { provider: 'openai', model: 'gpt-5.6-luna' },
  providers: [
    { provider: 'openai', label: 'OpenAI', configured: true, models: ['gpt-5.6-luna', 'gpt-4o'] },
    { provider: 'anvilgpt', label: 'AnvilGPT (Purdue RCAC)', configured: true,
      models: ['qwen3.8:27b'], ...anvil },
  ],
});

eq('a hidden model is unavailable, with the server\'s reason',
  unavailableChoice('qwen3:4b', 'anvilgpt', cat({ hidden: { 'qwen3:4b': REFUSED } })),
  `qwen3:4b is no longer offered: ${REFUSED}.`);
eq('found through `hidden` when no provider was saved with it',
  unavailableChoice('qwen3:4b', '', cat({ hidden: { 'qwen3:4b': REFUSED } }))?.startsWith('qwen3:4b'),
  true);
eq('a model the live list dropped (qwen3.6:27b on 2026-10-01) is unavailable',
  unavailableChoice('qwen3.6:27b', 'anvilgpt', cat({})),
  'qwen3.6:27b is no longer offered by AnvilGPT (Purdue RCAC).');
eq('an offered model is kept', unavailableChoice('qwen3.8:27b', 'anvilgpt', cat({})), null);
eq('the default ("") is kept', unavailableChoice('', '', cat({})), null);
eq('no catalogue yet: cannot tell, so kept', unavailableChoice('qwen3:4b', 'anvilgpt', null), null);
eq('a list that could not be fetched cannot rule a model out',
  unavailableChoice('qwen3.6:27b', 'anvilgpt', cat({ stale: true })), null);
eq('an unconfigured provider lists fallback ids, so it cannot either',
  unavailableChoice('qwen3.6:27b', 'anvilgpt', cat({ configured: false })), null);

eq('the group label counts what was hidden', hiddenNote({ 'qwen3:4b': REFUSED, 'qwen2.5:7b': REFUSED }),
  ' — 2 hidden: no tool calls');
eq('and says nothing when nothing was', hiddenNote(undefined), '');

if (bad) {
  console.log(`\n${bad} check(s) FAILED`);
  process.exit(1);
}
console.log('\nall model-choice checks passed');
