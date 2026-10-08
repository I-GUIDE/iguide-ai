/** A runnable check that the agent's own files are fetched with the caller's credentials, and
 * nobody else's are. `npm run check:files`.
 *
 * Same reasoning as check:auth and check:upload: there is no test runner here, and this fails
 * in a way no screenshot explains. A layer download sent without the key is refused in token
 * mode, the layer never appears, and the answer beside it says it is on the map. Measured on
 * agent.i-guide.io on 2026-10-04: four layer downloads in one turn, all 403 `not_signed_in`,
 * from a browser whose chat and upload calls had both gone through on the same key.
 */
import { readFileSync } from 'node:fs';
import { fetchAgentFile, type AgentConfig } from './agentClient';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

/** The last request, as the browser would have sent it. */
let seen: { url: string; init?: RequestInit } = { url: '' };
(globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown, init?: RequestInit) => {
  seen = { url: String(url), init };
  return { ok: true, status: 200 } as unknown as Response;
};
const setOrigin = (origin: string) => {
  (globalThis as unknown as { location: unknown }).location = { origin };
};
const key = () => (seen.init?.headers as Record<string, string> | undefined)?.['X-API-KEY'];

// Deployed: the UI and the API share an origin, and the endpoints are host-relative.
setOrigin('https://agent.i-guide.io');
const deployed = { endpoint: '/agent/chat/stream', uploadEndpoint: '/agent/files/upload',
                   apiKey: 'k-123' } as AgentConfig;
await fetchAgentFile('https://agent.i-guide.io/agent/files/file_1/download', deployed);
eq("the agent's own file carries the key", key(), 'k-123');
eq('  ...and the sign-in cookie', seen.init?.credentials, 'include');

await fetchAgentFile('https://tiles.example.org/layer.geojson', deployed);
eq('another host gets no key', key(), undefined);
eq('  ...and no cookie', seen.init?.credentials, undefined);

// A path that resolved to nothing comes through as '#'. Not a url, so nothing is attached.
await fetchAgentFile('#', deployed);
eq('a non-url gets no credentials', seen.init?.credentials, undefined);

// `npm run dev` against the deployed API: the UI is on localhost, the endpoints are absolute.
// The cookie has to be asked for here, because a cross-origin fetch leaves it behind by default.
setOrigin('http://localhost:5173');
const dev = { endpoint: 'https://agent.i-guide.io/agent/chat/stream',
              uploadEndpoint: 'https://agent.i-guide.io/agent/files/upload', apiKey: '' } as AgentConfig;
await fetchAgentFile('https://agent.i-guide.io/agent/files/file_2/download', dev);
eq('cross-origin dev still sends the cookie', seen.init?.credentials, 'include');
eq('  ...and no empty key header', key(), undefined);
await fetchAgentFile('http://localhost:5173/sample.geojson', dev);
eq("the dev server's own origin is not the agent's", seen.init?.credentials, undefined);

// Stop aborts every layer download the turn started, so the signal must survive the wrapper.
const ctl = new AbortController();
await fetchAgentFile('https://agent.i-guide.io/agent/files/file_3/download', dev,
                     { signal: ctl.signal });
eq('the abort signal is kept', seen.init?.signal === ctl.signal, true);

// The class, not the instance: three call sites in App.tsx had each fetched an agent file bare.
// npm runs this from the package root, so the source is read from there.
const bare = readFileSync('src/App.tsx', 'utf8').match(/\bfetch\(resolveUrl\(/g) || [];
eq('App.tsx fetches no agent file bare', bare.length, 0);

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
