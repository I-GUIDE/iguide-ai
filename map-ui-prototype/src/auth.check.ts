/** A runnable check for the refresh-and-retry contract. `npm run check:auth`.
 *
 * Same reasoning as check:fold — no test runner here, but this is logic that fails in ways a
 * screenshot cannot show. A retry that is not bounded is an infinite loop against the platform;
 * a refresh triggered on the wrong status is a loop against a token that will never validate;
 * a burst of parallel 401s that each refresh is a thundering herd on the auth backend. None of
 * those are visible until they are in production, so they get counted here.
 */
import { AuthError, authErrorFrom, isAuthStatus, setRefreshUrl, withTokenRetry } from './auth';

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

console.log(bad ? `\n${bad} FAILED` : '\nall passed');
process.exit(bad ? 1 : 0);
