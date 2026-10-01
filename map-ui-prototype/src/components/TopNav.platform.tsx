// THE ORIGINAL PROTOTYPE HEADER, kept so the platform-chrome look can be switched back to.
//
// Verbatim from commit 76dab0b (branch `prototype`), the state before issue #20 rebranded this
// deployment. Do not "tidy" it: its value is being an exact copy. The placeholders here are
// deliberate — Collections / Apps / Support / the search box were non-functional, mirroring the
// I-GUIDE platform chrome so the chat matched the platform when the map was hidden.
//
// Selected by VITE_UI_VARIANT=platform — see TopNav.tsx.
import { AccountBadge, accountNeedsAttention } from './AccountBadge';
import type { TopNavProps } from './TopNav';
// The reference I-GUIDE platform chrome. Nav items / search / account are PLACEHOLDERS
// (non-functional) so the chat matches the platform look when the map isn't shown.
export function TopNavPlatform(p: TopNavProps) {
  return (
    <header className="bar">
      <div className="bar-inner">
        <div className="brand">
          <svg className="iglogo" width="26" height="26" viewBox="0 0 100 100" aria-hidden="true">
            <polygon points="50,6 89,28 89,72 50,94 11,72 11,28" fill="none" stroke="#1aa37a" strokeWidth="9" strokeLinejoin="round" />
            <polygon points="50,26 71,38 71,62 50,74 29,62 29,38" fill="#a8b400" opacity="0.85" />
          </svg>
          <span className="brand-name">Knowledge Elements <span className="caret">▾</span></span>
        </div>
        <nav className="top">
          <span>Collections</span>
          <span>Apps <span className="caret">▾</span></span>
          <span>Support <span className="caret">▾</span></span>
        </nav>
        <div className="grow" />
        <div className="navsearch"><span className="mag">⌕</span><input placeholder="Search…" aria-label="Search (placeholder)" /></div>
        <button className="navbtn" title="Past conversations" onClick={p.onToggleHistory}>
          History{p.sessionCount ? ` (${p.sessionCount})` : ''}
        </button>
        {/* Gated the same way as the rs-embed header: in DEMO_MODE there is no key to enter
            and no endpoint worth changing. This variant is selected at BUILD time and Rollup
            drops it from the default bundle, so the gap was latent rather than live — but a
            platform build with DEMO_MODE on would have shown the dialog the switch exists to
            hide, and the two headers must not disagree about that. */}
        {!p.demoMode && (
          <button className="navbtn gear" title="Connection settings" onClick={p.onToggleSettings}>⚙</button>
        )}
        <span className="jpy" title="Jupyter (placeholder)">jpy</span>
        {/* The real account control, added for the same reason the gear gained a condition
            above: the two headers must not disagree about identity. The placeholder avatar
            stands down only when the badge has something to say — since the badge stopped
            rendering for a working account, asking `p.me` here would blank the avatar out of
            this header for everyone signed in, which is not a removal anybody asked for. */}
        {accountNeedsAttention(p.me)
          ? <AccountBadge me={p.me ?? null} />
          : <span className="avatar" title="Account (placeholder)" />}
      </div>
    </header>
  );
}
