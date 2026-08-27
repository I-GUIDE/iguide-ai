interface Props {
  onToggleSettings: () => void;
  onToggleHistory: () => void;
  sessionCount: number;
}

// The reference I-GUIDE platform chrome. Nav items / search / account are PLACEHOLDERS
// (non-functional) so the chat matches the platform look when the map isn't shown.
export function TopNav(p: Props) {
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
        <button className="navbtn gear" title="Connection settings" onClick={p.onToggleSettings}>⚙</button>
        <span className="jpy" title="Jupyter (placeholder)">jpy</span>
        <span className="avatar" title="Account (placeholder)" />
      </div>
    </header>
  );
}
