/**
 * Signed-in state, and what to do when the platform's access token ages out.
 *
 * The agent verifies the platform's JWT; it never mints or refreshes one. Refreshing is the
 * BROWSER's job: it calls the platform's own refresh endpoint, which validates the refresh
 * cookie and re-mints the `.i-guide.io` access cookie. Nothing here ever sees a token value —
 * both cookies are httpOnly, which is also why none of this reads `document.cookie`.
 *
 * The whole contract rests on one distinction the server makes for us:
 *
 *   401  the token was fine and has simply aged out  -> refresh once, retry once
 *   403  not signed in, or not permitted             -> stop and say so
 *
 * Collapsing those leaves this code unable to tell "refresh me" from "give up", so it would
 * either never refresh or refresh forever against a token that will never validate. Every
 * retry here is bounded at exactly one attempt for the same reason.
 */

/** Why a request was refused, as the server labels it. Never parsed out of prose. */
export type AuthReason =
  | 'token_expired'
  | 'not_signed_in'
  | 'token_invalid'
  | 'insufficient_role'
  | 'not_your_conversation'
  | 'identity_not_configured';

export class AuthError extends Error {
  readonly status: number;
  readonly reason: AuthReason | null;
  /** Present on insufficient_role, so the page can say what is actually required. The NAMES
   *  come from the server too: the scale is the platform's, lower is more privileged, and a
   *  copy of it here would be a copy that stops matching the day a tier is added. */
  readonly role?: number;
  readonly requiredRole?: number;
  readonly roleName?: string;
  readonly requiredRoleName?: string;

  constructor(status: number, reason: AuthReason | null, message: string,
              extra?: { role?: number; requiredRole?: number;
                        roleName?: string; requiredRoleName?: string }) {
    super(message);
    this.name = 'AuthError';
    this.status = status;
    this.reason = reason;
    this.role = extra?.role;
    this.requiredRole = extra?.requiredRole;
    this.roleName = extra?.roleName;
    this.requiredRoleName = extra?.requiredRoleName;
  }

  /** Recoverable by refreshing — the ONLY case worth retrying. */
  get isExpired(): boolean { return this.status === 401 || this.reason === 'token_expired'; }
}

/** Build an AuthError from a refused response, reading `reason` rather than the message text. */
export async function authErrorFrom(res: Response): Promise<AuthError> {
  let body: Record<string, unknown> = {};
  try { body = (await res.clone().json()) as Record<string, unknown>; } catch { /* not json */ }
  const reason = (typeof body.reason === 'string' ? body.reason : null) as AuthReason | null;
  const message = typeof body.error === 'string' && body.error
    ? body.error
    : res.status === 401 ? 'Your session has expired.' : 'Please sign in to use the agent.';
  return new AuthError(res.status, reason, message, {
    role: typeof body.role === 'number' ? body.role : undefined,
    requiredRole: typeof body.requiredRole === 'number' ? body.requiredRole : undefined,
    roleName: typeof body.roleName === 'string' ? body.roleName : undefined,
    requiredRoleName: typeof body.requiredRoleName === 'string' ? body.requiredRoleName : undefined,
  });
}

export function isAuthStatus(status: number): boolean {
  return status === 401 || status === 403;
}

/** Where the browser refreshes, as the deployment reports it. Empty until ui-config says. */
let refreshUrl = '';
export function setRefreshUrl(url: string | undefined | null): void {
  refreshUrl = (url || '').trim();
}
export function getRefreshUrl(): string { return refreshUrl; }

let signinUrl = '';
export function setSigninUrl(url: string | undefined | null): void {
  signinUrl = (url || '').trim();
}

/**
 * "the contributor role (4)", or just "role 6" when the platform's scale does not name it.
 *
 * The scale is SPARSE — 6, 7 and 9 are not roles — so an unnamed number is a real possibility
 * and must read as a number rather than being rounded to the nearest tier it is not.
 */
export function describeRole(role: number | undefined, name?: string): string {
  if (role === undefined) return name ? `the ${name.toLowerCase()} role` : 'a different role';
  return name ? `the ${name.toLowerCase()} role (${role})` : `role ${role}`;
}

/** The same fact as a LABEL rather than a clause: "Contributor (4)".
 *  describeRole reads correctly inside a sentence and badly in a key/value row, which is
 *  where it first went — "Role: the contributor role (4)". */
export function roleLabel(role: number | undefined, name?: string): string {
  if (role === undefined) return name || 'Unknown';
  return name ? `${name} (${role})` : `Role ${role}`;
}

/**
 * What to tell the person. Each case is a DIFFERENT thing to do about it, which is the reason
 * the server labels refusals instead of just refusing:
 *
 *   not signed in       -> sign in; fixable in one click
 *   insufficient role   -> signing in again will not help; ask someone for access
 *   wrong conversation  -> signed in fine, but this link is not theirs
 *
 * Because the role gate starts at contributor, "insufficient role" is what MOST platform
 * accounts will hit — an ordinary trusted user is refused — so it is the message that has to
 * name what is required rather than saying "forbidden" and leaving them nowhere to go.
 */
export function authMessage(err: AuthError): string {
  const signIn = signinUrl ? ` [Sign in](${signinUrl}) and try again.` : ' Please sign in and try again.';
  switch (err.reason) {
    case 'insufficient_role':
      return 'Your I-GUIDE account is signed in, but does not have access to the agent'
        + (err.requiredRole !== undefined
          ? ` — it needs ${describeRole(err.requiredRole, err.requiredRoleName)} or above${
            err.role !== undefined
              ? `, and this account is ${describeRole(err.role, err.roleName)}` : ''}.`
          : '.')
        + ' Signing in again will not change this; ask an I-GUIDE administrator for access.';
    case 'not_your_conversation':
      return 'That conversation belongs to a different account.';
    case 'identity_not_configured':
      return 'This deployment cannot verify sign-ins right now. That is a server-side problem, '
        + 'not something you can fix — please report it.';
    case 'token_expired':
    case 'not_signed_in':
    case 'token_invalid':
      return `You are not signed in.${signIn}`;
    default:
      // No identity `reason` at all, so this is NOT an identity refusal — an API-key rejection
      // arrives as a bare 403. Reporting it as "not signed in" sends someone to a login page
      // that cannot fix it; observed live, after signing in successfully.
      return err.message;
  }
}

/** In-flight refresh, shared so a burst of 401s makes ONE call rather than one each.
 *  Four parallel requests expiring together would otherwise race to refresh, and the losers
 *  would retry against a cookie that was replaced out from under them mid-flight. */
let inFlight: Promise<boolean> | null = null;

/**
 * Ask the platform for a fresh access cookie. Resolves true if it set one.
 *
 * `credentials: 'include'` is required and not optional: the refresh cookie is what
 * authenticates this call, and the platform backend is a different ORIGIN from the agent even
 * though it is the same site. The backend must also list this origin in ALLOWED_DOMAIN_LIST or
 * the browser blocks the response before this code sees it.
 */
export function refreshAccessToken(): Promise<boolean> {
  if (!refreshUrl) return Promise.resolve(false);
  if (inFlight) return inFlight;
  const attempt = (async () => {
    try {
      const res = await fetch(refreshUrl, { method: 'POST', credentials: 'include' });
      return res.ok;
    } catch {
      return false;      // offline, blocked by CORS, backend down — all "could not refresh"
    }
  })();
  inFlight = attempt;
  // Cleared the moment it SETTLES, not on a timer. Everyone who asked while it was running
  // shares this one result, which is the point; but a 401 arriving after it finished is newer
  // than the refresh and must trigger its own. Holding the result any longer pins the client to
  // a stale answer — including a stale `true`, which retries forever against a cookie that was
  // never actually replaced.
  attempt.then(() => { inFlight = null; }, () => { inFlight = null; });
  return attempt;
}

/**
 * Run a request; on an expired token, refresh once and run it exactly once more.
 *
 * `run` must be a thunk rather than a Promise: a retry has to ISSUE a new request, and an
 * already-started Promise cannot be re-issued. Anything that is not an expiry — not signed in,
 * insufficient role, someone else's conversation — is rethrown untouched, because no number of
 * refreshes will fix it.
 */
export async function withTokenRetry<T>(run: () => Promise<T>): Promise<T> {
  try {
    return await run();
  } catch (err) {
    if (!(err instanceof AuthError) || !err.isExpired) throw err;
    const refreshed = await refreshAccessToken();
    if (!refreshed) {
      // Refresh itself failed: the refresh token is gone or expired too. This is a sign-in,
      // not another retry — retrying here is how a loop starts.
      throw new AuthError(403, 'not_signed_in', 'Your session has expired. Please sign in again.');
    }
    return await run();
  }
}
