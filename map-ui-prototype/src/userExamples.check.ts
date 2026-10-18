// The client half of user-authored examples (architecture stage 73): the draft the editor keeps,
// the body it sends, and the share link. The server's rules are tested in
// rag_pipeline/tests/test_user_examples.py.
import { DEMOS, demosByCategory, nextStep } from './demos';
import { type ExampleDraft, type UserExample, draftFrom, draftProblem, emptyDraft, exampleForm, isUserExample,
  moveStep, parseTags, shareLink, shareTokenFrom, withoutShareToken } from './userExamples';

let failed = 0;
const ok = (cond: unknown, msg: string) => { if (!cond) { failed++; console.error('FAIL', msg); } };

const steps = [{ label: 'a', prompt: 'A?' }, { label: 'b', prompt: 'B?' }, { label: 'c', prompt: 'C?' }];
ok(moveStep(steps, 1, -1).map((s) => s.label).join('') === 'bac', 'a step moves up');
ok(moveStep(steps, 1, 1).map((s) => s.label).join('') === 'acb', 'a step moves down');
ok(moveStep(steps, 0, -1) === steps && moveStep(steps, 2, 1) === steps, 'the ends stay put');
ok(steps.map((s) => s.label).join('') === 'abc', 'moving does not change the list it was given');

ok(parseTags(' buffer, overlay ,, buffer,census ').join('|') === 'buffer|overlay|census', 'tags are trimmed and deduplicated');

const draft: ExampleDraft = { ...emptyDraft('Measurement'), title: 'Hospitals near rivers',
  steps: [{ label: 'Load', prompt: 'Load the hospitals.' }, { label: '', prompt: '' }] };
ok(draftProblem(draft, []) === null, 'a trailing blank step is dropped, not refused');
ok(draftProblem({ ...draft, title: ' ' }, [])?.includes('title'), 'a title is required');
ok(draftProblem({ ...draft, steps: [{ label: 'x', prompt: '' }] }, [])?.includes('Step 1'), 'a labelled step needs a question');
ok(draftProblem({ ...draft, steps: [] }, [])?.includes('at least one step'), 'one step is required');
const limits = { maxExamples: 50, maxSteps: 1, maxFiles: 1, maxBytes: 10 };
ok(draftProblem({ ...draft, steps: [steps[0], steps[1]] }, [], limits)?.includes('1 steps'), 'the step limit is checked');
const f = (name: string, n: number) => new File([new Uint8Array(n)], name);
ok(draftProblem({ ...draft, keepFiles: ['a.csv'] }, [f('b.csv', 1)], limits)?.includes('1 files'), 'kept + new files count together');
ok(draftProblem(draft, [f('a.csv', 1), f('a.csv', 1)], { ...limits, maxFiles: 1 }) === null, 'a replaced file is one file');
ok(draftProblem(draft, [f('big.tif', 11)], limits)?.includes('MB'), 'the size limit is checked');

const form = exampleForm({ ...draft, tags: 'x, y', keepFiles: ['old.csv'] }, [f('new.csv', 3)], true);
const body = JSON.parse(String(form.get('example')));
ok(body.title === 'Hospitals near rivers' && body.category === 'Measurement', 'the definition is sent as JSON');
ok(body.steps.length === 1 && body.steps[0].prompt === 'Load the hospitals.', 'blank steps are not sent');
ok(body.tags.join('|') === 'x|y' && body.keepFiles.join() === 'old.csv', 'tags as a list, and the files to keep on an edit');
ok((form.getAll('files') as File[]).map((x) => x.name).join() === 'new.csv', 'new files go under `files`');
ok(!('keepFiles' in JSON.parse(String(exampleForm(draft, [], false).get('example')))), 'a new example sends no keepFiles');

const ex: UserExample = {
  id: 'ex_0123456789abcdef', source: 'user', title: 'Mine', blurb: 'About it', category: 'Something new',
  tags: ['t1', 't2'], steps: [{ label: 'One', prompt: 'One?' }], mine: true, visibility: 'link', published: false,
  files: [{ name: 'a.csv', bytes: 3, url: '/agent/examples/ex_0123456789abcdef/files/a.csv' }], shareToken: 'tok',
};
const back = draftFrom(ex);
ok(back.tags === 't1, t2' && back.description === 'About it' && back.keepFiles.join() === 'a.csv', 'an example round-trips into a draft');
back.steps[0].label = 'changed';
ok(ex.steps[0].label === 'One', 'editing the draft does not touch the example');
ok(isUserExample(ex) && !isUserExample(DEMOS[0]), 'user examples are told apart from built-in ones');

// A published example is filed beside the built-in ones; an unknown category still shows.
const groups = demosByCategory([...DEMOS, ex]);
ok(groups[groups.length - 1].category === 'Something new' && groups[groups.length - 1].cases[0] === ex,
   'a user example with a new category is listed under it, last');
ok(nextStep(ex, new Set()) === 0, 'a user example steps like a built-in one');

const link = shareLink('AbCdEf_123-xyz', 'https://agent.example.org/map/?model=x#top');
ok(link === 'https://agent.example.org/map/?example=AbCdEf_123-xyz', `the link is the page itself with only the token: ${link}`);
ok(shareTokenFrom(link) === 'AbCdEf_123-xyz', 'the token is read back from the link');
ok(shareTokenFrom('https://h/?example=<script>') === null, 'a malformed token is ignored');
ok(shareTokenFrom('https://h/') === null, 'no token, no example');
ok(withoutShareToken('https://h/p?a=1&example=AbCdEf_123') === 'https://h/p?a=1', 'the token leaves the address once handled');

if (failed) { console.error(`${failed} user-example check(s) failed`); process.exit(1); }
console.log('user examples: all checks passed');
