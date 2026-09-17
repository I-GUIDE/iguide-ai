import { useEffect, useRef, useState } from 'react';

import { describeRole, roleLabel } from '../auth';
import type { WhoAmI } from '../agentClient';

/**
 * Who you are, in the header — and, when you are nobody, a way to become somebody.
 *
 * This exists because of a shape the identity work kept producing: the page could only tell
 * you about your access AFTER you had typed something and been refused. In token mode a
 * signed-out visitor looked identical to a signed-in one — an empty history, an open composer —
 * and found out where they stood by losing a question to a 403. The refusal messages were
 * careful and specific, but they arrived at the worst possible moment.
 *
 * So the badge reports the state BEFORE it costs anything, and the three states are three
 * different things to do about it:
 *
 *   signed out          -> sign in; one click, and the link comes from the server
 *   signed in, refused  -> signing in again will not help; ask an administrator
 *   signed in, allowed  -> nothing to do; just say who the server thinks you are
 *
 * Rendered only in token mode. Dev and demo deployments identify nobody, so a badge there
 * would be an account control for an account that does not exist.
 */
export function AccountBadge({ me }: { me: WhoAmI | null }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // Null means the whoami call has not landed (or failed). Rendering "signed out" for that
  // would accuse a signed-in visitor of being signed out for as long as the request takes,
  // and a flash of "Sign in" on every load is worse than a beat of nothing.
  if (!me) return null;

  if (!me.signedIn) {
    const href = me.signinUrl || '';
    return href
      ? <a className="navbtn acct acct-out" href={href} title="Sign in to the I-GUIDE platform">Sign in</a>
      // No sign-in URL means the deployment could not resolve one — a server-side problem the
      // visitor cannot act on, so it says so rather than offering a link to nowhere.
      : <span className="navbtn acct acct-out" title={me.reason || undefined}>Not signed in</span>;
  }

  const label = roleLabel(me.user?.role, me.user?.roleName ?? undefined);
  const inSentence = describeRole(me.user?.role, me.user?.roleName ?? undefined);

  return (
    <div className="acctwrap" ref={wrap}>
      <button type="button" aria-expanded={open} aria-haspopup="dialog"
        className={`navbtn acct ${me.permitted ? 'acct-ok' : 'acct-blocked'}`}
        title={me.permitted ? `Signed in as ${inSentence}` : 'Signed in, but without access to the agent'}
        onClick={() => setOpen((v) => !v)}>
        {me.permitted ? (me.user?.roleName || label) : 'No access'}
      </button>
      {open && (
        <div className="acctcard" role="dialog" aria-label="Account">
          <div className="acctrow">
            <span className="acctkey">Account</span>
            {/* The platform's user id, not an email: it is what the token carries and what
                every owned file and conversation on this deployment is stamped with, so it is
                the identifier worth being able to quote when asking about access. */}
            <code className="acctval">{me.user?.id}</code>
          </div>
          <div className="acctrow">
            <span className="acctkey">Role</span>
            <span className="acctval">{label}</span>
          </div>
          {me.platformTier && (
            <div className="acctrow">
              <span className="acctkey">Platform</span>
              <span className="acctval">{me.platformTier}</span>
            </div>
          )}
          {!me.permitted && (
            <p className="acctnote">
              This account is signed in but cannot use the agent
              {me.requiredRole != null
                ? `: it needs ${describeRole(me.requiredRole, me.requiredRoleName ?? undefined)} or above.`
                : '.'}{' '}
              Signing in again will not change this — ask an I-GUIDE administrator for access.
            </p>
          )}
          <p className="acctnote acctnote-quiet">
            Your conversations and files on this deployment belong to this account, and follow
            it to another browser.
          </p>
        </div>
      )}
    </div>
  );
}
