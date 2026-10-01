import { useEffect, useLayoutEffect, useRef, useState } from 'react';

import { AccountBadge } from './AccountBadge';
import { IGuideMark } from './IGuideMark';
import { TopNavPlatform } from './TopNav.platform';
import { isPlatformVariant, type AppTab } from '../uiVariant';
import type { WhoAmI } from '../agentClient';

export interface TopNavProps {
  /** Server-side DEMO_MODE: the deployment needs no key, so there is nothing to configure. */
  demoMode?: boolean;
  /** Who the server says we are, in TOKEN mode only — null in dev and demo, which identify
   *  nobody, and null until the answer lands. */
  me?: WhoAmI | null;
  onToggleSettings: () => void;
  onToggleHistory: () => void;
  sessionCount: number;
  tab: AppTab;
  onSetTab: (t: AppTab) => void;
}

const TABS: { id: AppTab; label: string; title: string }[] = [
  { id: 'chat', label: 'Chat', title: 'Ask anything — the map opens when an answer needs it' },
  { id: 'rs', label: 'rs-embed demo', title: 'Draw a region and run satellite-embedding operations on it' },
];

// The header for the rs-embed deployment (issue #20). This used to mirror the I-GUIDE platform
// chrome with non-functional placeholders — Collections / Apps / Support / a search box — which
// made the page look like the platform without behaving like it: every one of them was dead on
// click. They are gone, and the one link that DOES go somewhere replaces them.
//
// The MARK is the link back to the platform, so there is no separate text link. Only History
// and the settings gear remain on the right: the jpy badge and the account avatar were platform
// placeholders that did nothing here.
function TopNavRsEmbed(p: TopNavProps) {
  // One travelling lens rather than a background that blinks from one button to the other. The
  // two labels are very different widths — "Chat" against "rs-embed demo" — so the lens resizes
  // as it moves, which is where most of the liquid character comes from; it also stretches along
  // the way and settles, the way a drop of glass would.
  const navRef = useRef<HTMLElement>(null);
  const [lens, setLens] = useState<{ x: number; w: number } | null>(null);
  const [travelling, setTravelling] = useState(false);
  const mounted = useRef(false);

  useLayoutEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const measure = () => {
      const on = nav.querySelector<HTMLButtonElement>('.tab.on');
      if (on) setLens({ x: on.offsetLeft, w: on.offsetWidth });
    };
    measure();
    // Webfonts land after first paint and the labels reflow with them, so a lens measured once
    // keeps whatever width the fallback face happened to give it. The observer also covers the
    // window being narrowed.
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(nav);
    return () => ro.disconnect();
  }, [p.tab]);

  useEffect(() => {
    // Not on the first render: there is no travel to animate when the page opens, and a lens
    // that stretches on load reads as a glitch rather than as a material.
    if (!mounted.current) { mounted.current = true; return; }
    setTravelling(true);
    const id = setTimeout(() => setTravelling(false), 460);
    return () => clearTimeout(id);
  }, [p.tab]);

  return (
    <header className="bar">
      <div className="bar-inner">
        <div className="brand">
          <a className="marklink" href="https://platform.i-guide.io" target="_blank"
             rel="noopener noreferrer" aria-label="Back to the I-GUIDE Platform"
             title="Back to the I-GUIDE Platform">
            <IGuideMark className="iglogo" />
          </a>
          <span className="brand-name">I-GUIDE AI</span>
        </div>
        {/* A demo surface, not a second app: the tabs choose what the page is SET UP for, and
            the conversation carries across both. */}
        <nav className="tabs" role="tablist" aria-label="Workspace" ref={navRef}>
          {/* Rendered only once measured, so it never flashes at the wrong width. Decorative:
              the selected state is on the buttons, where a screen reader reads it. */}
          {lens && (
            /* Real values, not custom properties: a transition on a width/transform whose
               value comes from an unregistered var() does not re-resolve when the var changes.
               Chrome kept rendering the previous tab's 133px while --tab-w already read 57px,
               so the lens never moved. Nothing needed them either — travel and stretch live on
               two elements, so this transform is only ever a translate. */
            <span className={`tabglass${travelling ? ' travelling' : ''}`} aria-hidden="true"
                  style={{ transform: `translateX(${lens.x}px)`, width: `${lens.w}px` }}>
              {/* Keyed on the tab so a second click restarts the stretch instead of finding the
                  class already applied and doing nothing. The travel itself is on the parent, so
                  remounting this does not interrupt it. */}
              <span key={p.tab} className="tabglass-lens" />
            </span>
          )}
          {TABS.map((t) => (
            <button key={t.id} role="tab" type="button" title={t.title}
              aria-selected={p.tab === t.id}
              className={`tab ${p.tab === t.id ? 'on' : ''}`}
              onClick={() => p.onSetTab(t.id)}>{t.label}</button>
          ))}
        </nav>
        <div className="grow" />
        <button className="navbtn" title="Past conversations" onClick={p.onToggleHistory}>
          History{p.sessionCount ? ` (${p.sessionCount})` : ''}
        </button>
        <AccountBadge me={p.me ?? null} />
        {/* In demo mode there is no key to enter and no endpoint worth changing, and the one
            control the dialog still offered — the mock/live switch — is not what an audience
            should find first. Hidden rather than disabled: a greyed gear invites a click that
            explains nothing. */}
        {!p.demoMode && (
          <button className="navbtn gear" title="Connection settings" onClick={p.onToggleSettings}>⚙</button>
        )}
      </div>
    </header>
  );
}

// The original prototype page is kept verbatim in TopNav.platform.tsx and selected here, so
// switching back is a build flag rather than a revert. See src/uiVariant.ts.
export function TopNav(p: TopNavProps) {
  return isPlatformVariant ? <TopNavPlatform {...p} /> : <TopNavRsEmbed {...p} />;
}
