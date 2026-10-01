/** A runnable check for the refresh-and-retry contract. `npm run check:auth`.
 *
 * Same reasoning as check:fold — no test runner here, but this is logic that fails in ways a
 * screenshot cannot show. A retry that is not bounded is an infinite loop against the platform;
 * a refresh triggered on the wrong status is a loop against a token that will never validate;
 * a burst of parallel 401s that each refresh is a thundering herd on the auth backend. None of
 * those are visible until they are in production, so they get counted here.
 */
import { AuthError, authErrorFrom, authMessage, describeRole, isAuthStatus, setRefreshUrl,
  withTokenRetry } from './auth';
import { visibleTo } from './sessionStore';
import { fetchWhoAmI, type AgentConfig } from './agentClient';

let bad = 0;
const eq = (label: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: got ${JSON.stringify(got)}`);
};

/** Count every call the browser would actually make. */
let refreshCalls = 0;
let refreshSucceeds = true;
(globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown, init?: unknown) => {
  void url; void init;
  refreshCalls++;
  return { ok: refreshSucceeds } as Response;
};
setRefreshUrl('https://backend-dev.i-guide.io/api/refresh-token');

const refused = (status: number, reason: string, extra: Record<string, unknown> = {}) =>
  ({ status, clone: () => ({ json: async () => ({ reason, error: 'nope', ...extra }) }) }) as unknown as Response;

const reset = () => { refreshCalls = 0; refreshSucceeds = true; };

// --- the happy path: one refusal, one refresh, one retry -------------------------
await (async () => {
  reset();
  let attempts = 0;
  const got = await withTokenRetry(async () => {
    attempts++;
    if (attempts === 1) throw await authErrorFrom(refused(401, 'token_expired'));
    return 'answered';
  });
  eq('an expired token refreshes and retries', got, 'answered');
  eq('  ...issuing exactly one retry', attempts, 2);
  eq('  ...and exactly one refresh', refreshCalls, 1);
})();

// --- the bound: a persistently expired token must not loop -----------------------
await (async () => {
  reset();
  let attempts = 0;
  let caught: string | null = null;
  try {
    await withTokenRetry(async () => {
      attempts++;
      throw await authErrorFrom(refused(401, 'token_expired'));
    });
  } catch (err) { caught = (err as AuthError).reason; }
  eq('a token that stays expired stops after one retry', attempts, 2);
  eq('  ...refreshing only once', refreshCalls, 1);
  eq('  ...and surfacing the second refusal', caught, 'token_expired');
})();

// --- a failed refresh is a sign-in, not another retry ----------------------------
await (async () => {
  reset();
  refreshSucceeds = false;
  let attempts = 0;
  let reason: string | null = null;
  try {
    await withTokenRetry(async () => {
      attempts++;
      throw await authErrorFrom(refused(401, 'token_expired'));
    });
  } catch (err) { reason = (err as AuthError).reason; }
  eq('a failed refresh does not retry the request', attempts, 1);
  eq('  ...it asks for a sign-in', reason, 'not_signed_in');
})();

// --- what must NEVER refresh ------------------------------------------------------
for (const [reason, status] of [['not_signed_in', 403], ['token_invalid', 403],
                                ['insufficient_role', 403], ['not_your_conversation', 404]] as const) {
  await (async () => {
    reset();
    let attempts = 0;
    let seen: string | null = null;
    try {
      await withTokenRetry(async () => {
        attempts++;
        throw await authErrorFrom(refused(status, reason));
      });
    } catch (err) { seen = (err as AuthError).reason; }
    eq(`${reason} is not retried`, attempts, 1);
    eq(`  ...and triggers no refresh`, refreshCalls, 0);
    eq(`  ...rethrown untouched`, seen, reason);
  })();
}

// --- a non-auth failure passes straight through -----------------------------------
await (async () => {
  reset();
  let message = '';
  try {
    await withTokenRetry(async () => { throw new Error('HTTP 500: agent exploded'); });
  } catch (err) { message = (err as Error).message; }
  eq('an ordinary error is not mistaken for an expiry', message, 'HTTP 500: agent exploded');
  eq('  ...and triggers no refresh', refreshCalls, 0);
})();

// --- a burst of expiries shares ONE refresh ---------------------------------------
await (async () => {
  reset();
  const attempts = [0, 0, 0, 0];
  await Promise.all(attempts.map((_, i) => withTokenRetry(async () => {
    attempts[i]++;
    if (attempts[i] === 1) throw await authErrorFrom(refused(401, 'token_expired'));
    return i;
  })));
  eq('four requests expiring together each retry once', attempts, [2, 2, 2, 2]);
  // Without sharing, the losers would retry against a cookie replaced mid-flight.
  eq('  ...but share a single refresh', refreshCalls, 1);
})();

// --- the detail the page needs to explain itself -----------------------------------
await (async () => {
  const err = await authErrorFrom(refused(403, 'insufficient_role', { role: 8, requiredRole: 4 }));
  eq('an insufficient role carries the numbers', [err.role, err.requiredRole], [8, 4]);
  eq('  ...and is not treated as expired', err.isExpired, false);
  eq('a bare 401 counts as expired even with no reason',
     (await authErrorFrom({ status: 401, clone: () => ({ json: async () => ({}) }) } as unknown as Response)).isExpired,
     true);
  eq('401 and 403 are the auth statuses', [isAuthStatus(401), isAuthStatus(403), isAuthStatus(500)],
     [true, true, false]);
})();

// --- a refusal that is not about identity must not read like one -------------------
await (async () => {
  // The live bug: token mode still demanded the API key, the browser had none, and the bare
  // 403 that came back was rendered as "You are not signed in" — sending someone who HAD just
  // signed in to the login page again.
  const bare = await authErrorFrom({ status: 403,
    clone: () => ({ json: async () => ({ error: 'Forbidden: invalid API key.' }) }) } as unknown as Response);
  eq('a 403 with no reason keeps the server\'s own message',
     authMessage(bare), 'Forbidden: invalid API key.');
  eq('  ...and is not reported as a sign-in problem',
     authMessage(bare).includes('not signed in'), false);
  const real = await authErrorFrom({ status: 403,
    clone: () => ({ json: async () => ({ reason: 'not_signed_in', error: 'Please sign in.' }) }) } as unknown as Response);
  eq('a real identity refusal still says so', authMessage(real).includes('not signed in'), true);
})();

// --- roles are NAMED by the server, never by a copy of the scale kept here -----------
await (async () => {
  // The platform's scale runs backwards and is sparse: 1 is the most privileged and 6, 7 and 9
  // are not roles at all. The client cannot derive any of that, so the names ride along with
  // the numbers on the refusal — and an unnamed number has to read as a number rather than be
  // rounded to a tier it is not.
  eq('a named role reads as a name and a number', describeRole(4, 'Contributor'),
     'the contributor role (4)');
  eq('an unnamed role reads as a bare number', describeRole(6), 'role 6');

  const named = await authErrorFrom(refused(403, 'insufficient_role',
    { role: 8, requiredRole: 4, roleName: 'Trusted user', requiredRoleName: 'Contributor' }));
  eq('the refusal carries both names', [named.roleName, named.requiredRoleName],
     ['Trusted user', 'Contributor']);
  const message = authMessage(named);
  eq('  ...and the message names what is required', message.includes('the contributor role (4)'), true);
  eq('  ...and what this account is', message.includes('the trusted user role (8)'), true);
  eq('  ...and says signing in again will not help',
     message.includes('Signing in again will not change this'), true);

  // A server that has not been updated yet sends numbers only. The message must still work.
  const unnamed = await authErrorFrom(refused(403, 'insufficient_role', { role: 8, requiredRole: 4 }));
  eq('numbers alone still produce a usable message',
     authMessage(unnamed).includes('it needs role 4 or above'), true);
})();

// --- a cache must not serve the last user's conversations ---------------------------
await (async () => {
  // `null` viewer means two different things and the filter used to conflate them: "this
  // deployment identifies nobody" (dev/demo, where nothing is stamped) versus "nobody is
  // signed in yet" (token mode, where records ARE stamped). Found live — History (27) sat
  // beside a Sign in button on the deployment's first minutes in token mode.
  const mine = { ownerId: 'u-1' };
  const theirs = { ownerId: 'u-2' };
  const unowned = { ownerId: null };
  eq('my own record is mine', visibleTo(mine, 'u-1'), true);
  eq('someone else\'s is not', visibleTo(theirs, 'u-1'), false);
  eq('an unowned record is shared', visibleTo(unowned, 'u-1'), true);
  eq('  ...including in dev, where there is no viewer at all', visibleTo(unowned, null), true);
  eq('an OWNED record is hidden from a signed-out viewer', visibleTo(mine, null), false);
  eq('  ...and from an undefined one', visibleTo(mine, undefined), false);
})();

// --- an aged-out access cookie must not present as signed out ------------------------
await (async () => {
  // withTokenRetry cannot cover this: it reacts to a 401, and /agent/whoami answers 200 by
  // design because it exists to EXPLAIN a refusal rather than make one. So an access cookie
  // aging out (one hour) turned a signed-in page into a signed-out one — badge reading
  // "Sign in", history emptied — while the refresh cookie sat unused. Observed live.
  const cfg = { endpoint: 'https://agent.example/agent/chat/stream', uploadEndpoint: '',
                apiKey: '' } as AgentConfig;
  let whoamiCalls = 0, refreshes = 0, refreshWorks = true;
  const expired = { mode: 'token', signedIn: false, user: null, permitted: false,
                    reason: 'TokenExpired: access token expired', reasonCode: 'token_expired' };
  const live = { mode: 'token', signedIn: true, user: { id: 'u-1', role: 4, roleName: 'Contributor' },
                 permitted: true, reason: null, reasonCode: null };
  (globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown) => {
    const u = String(url);
    if (u.includes('/agent/whoami')) {
      whoamiCalls++;
      const body = (whoamiCalls === 1 || !refreshWorks) ? expired : live;
      return { ok: true, json: async () => body } as unknown as Response;
    }
    refreshes++;
    return { ok: refreshWorks } as Response;
  };

  let me = await fetchWhoAmI(cfg);
  eq('an expired token is refreshed rather than reported', me?.signedIn, true);
  eq('  ...with exactly one refresh', refreshes, 1);
  eq('  ...and exactly one re-ask', whoamiCalls, 2);

  // Bounded at one attempt: a refresh that "succeeds" but yields another expired token must
  // not loop the page against the auth backend.
  whoamiCalls = 0; refreshes = 0; refreshWorks = true;
  const stillExpired = async (url: unknown) => {
    const u = String(url);
    if (u.includes('/agent/whoami')) { whoamiCalls++; return { ok: true, json: async () => expired } as unknown as Response; }
    refreshes++; return { ok: true } as Response;
  };
  (globalThis as unknown as { fetch: unknown }).fetch = stillExpired;
  me = await fetchWhoAmI(cfg);
  eq('a refresh that does not help is not retried', [refreshes, whoamiCalls], [1, 2]);
  eq('  ...and the caller still learns it is expired', me?.reasonCode, 'token_expired');

  // A failed refresh must not cost a second whoami either.
  whoamiCalls = 0; refreshes = 0; refreshWorks = false;
  (globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown) => {
    const u = String(url);
    if (u.includes('/agent/whoami')) { whoamiCalls++; return { ok: true, json: async () => expired } as unknown as Response; }
    refreshes++; return { ok: false } as Response;
  };
  me = await fetchWhoAmI(cfg);
  eq('a failed refresh does not re-ask', [refreshes, whoamiCalls], [1, 1]);

  // Only expiry. A forged token or an under-privileged account is not fixed by a new cookie.
  for (const code of ['token_invalid', 'insufficient_role', 'not_signed_in']) {
    whoamiCalls = 0; refreshes = 0;
    (globalThis as unknown as { fetch: unknown }).fetch = async (url: unknown) => {
      const u = String(url);
      if (u.includes('/agent/whoami')) { whoamiCalls++; return { ok: true, json: async () => ({ ...expired, reasonCode: code }) } as unknown as Response; }
      refreshes++; return { ok: true } as Response;
    };
    await fetchWhoAmI(cfg);
    eq(`${code} triggers no refresh`, [refreshes, whoamiCalls], [0, 1]);
  }
})();

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
