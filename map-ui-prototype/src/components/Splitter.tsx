import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, KeyboardEvent as ReactKeyboardEvent } from 'react';

/** How far one arrow key moves the seam; Shift multiplies it. */
const STEP_PX = 16;
const STEP_FAST_PX = 64;

interface Props {
  /** Floor for the chat pane, in px. */
  chatMin: number;
  /** Floor for the MAP pane, in px. Never 0: a zero-width map is unmounted, not collapsed. */
  mapMin: number;
  /** What double-click and Enter return to. */
  defaultWidth: number;
  onResizeStart: () => void;
  /** Fires ONCE per gesture — on release, on a key step, on reset. Never per frame. */
  onResizeEnd: (chatW: number) => void;
}

/**
 * The seam between the map and the chat.
 *
 * The width is written straight to the DOM as `--chat-w` while the pointer is down, and only
 * committed to React state on release. A width in state would re-render the whole transcript on
 * every pointermove — nothing in this app is memoized, and each turn re-renders its answer HTML —
 * on the very frames MapLibre is reallocating its WebGL buffer.
 */
export function Splitter({ chatMin, mapMin, defaultWidth, onResizeStart, onResizeEnd }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ id: number; x0: number; w0: number; wsW: number; handle: number } | null>(null);
  const next = useRef(0);
  const raf = useRef(0);
  const [dragging, setDragging] = useState(false);
  // The seam's position as a percentage of the workspace measured from the LEFT, so the value
  // grows as the seam moves right — which is what "increase" means to a screen reader. That
  // makes the announced value the MAP's share; aria-valuetext spells out both panes.
  const [val, setVal] = useState({ now: 50, min: 0, max: 100 });

  const workspace = () => ref.current?.closest('.workspace') as HTMLElement | null;
  const chatPane = () => workspace()?.querySelector('.chat') as HTMLElement | null;
  const setWidth = (px: number) => { workspace()?.style.setProperty('--chat-w', `${px}px`); };

  // The handle's own width comes from the layout, not a constant here: styles.css owns it.
  const clampW = useCallback(
    (px: number, wsW: number, handle: number) =>
      Math.round(Math.min(Math.max(px, chatMin), Math.max(chatMin, wsW - mapMin - handle))),
    [chatMin, mapMin],
  );

  const publish = useCallback((chatW: number, wsW: number, handle: number) => {
    const usable = wsW - handle;
    if (usable <= 0) return;
    const pct = (n: number) => Math.round((n / usable) * 100);
    setVal({ now: pct(usable - chatW), min: pct(mapMin), max: pct(usable - chatMin) });
  }, [chatMin, mapMin]);

  /** Read the announced value back off the layout — on mount, and when the window changes it. */
  const measure = useCallback(() => {
    const ws = workspace(), el = ref.current, chat = chatPane();
    if (ws && el && chat) publish(chat.offsetWidth, ws.clientWidth, el.offsetWidth);
  }, [publish]);

  useLayoutEffect(() => {
    measure();
    const ws = workspace();
    if (!ws || typeof ResizeObserver === 'undefined') return;
    // styles.css does the actual re-clamping, live, inside clamp() — this only re-reads the
    // result so what is announced still matches what is on screen after a window resize, a
    // browser-zoom change, a rotation, or devtools docking.
    const ro = new ResizeObserver(measure);
    ro.observe(ws);
    return () => ro.disconnect();
  }, [measure]);

  const end = useCallback((commit: boolean) => {
    const d = drag.current;
    if (!d) return;                       // every exit path lands here; only the first counts
    drag.current = null;
    setDragging(false);
    if (raf.current) { cancelAnimationFrame(raf.current); raf.current = 0; }
    // Throws NotFoundError if the pointer is already gone. The drag is over either way.
    try { ref.current?.releasePointerCapture(d.id); } catch { /* */ }
    const w = commit ? next.current : d.w0;
    if (!commit) setWidth(w);             // Escape puts it back where the drag started
    publish(w, d.wsW, d.handle);
    onResizeEnd(w);
  }, [onResizeEnd, publish]);
  // Read from listeners and from unmount, both of which would otherwise close over a stale copy.
  const endRef = useRef(end); endRef.current = end;

  useEffect(() => {
    if (!dragging) return;
    // Pointer capture normally guarantees a pointerup or pointercancel, but a drag that is
    // alt-tabbed away or interrupted by a system gesture can leave the pane following a pointer
    // nobody is pressing. StrictMode re-runs this; the removals make that harmless.
    const bail = () => endRef.current(true);
    window.addEventListener('blur', bail);
    document.addEventListener('visibilitychange', bail);
    return () => {
      window.removeEventListener('blur', bail);
      document.removeEventListener('visibilitychange', bail);
    };
  }, [dragging]);
  // The handle can be taken out from under the pointer — a window narrowed past the two-pane
  // minimum folds to one pane — and the .resizing class lives on .app, so the drag has to be
  // endable from here too or the whole app keeps a col-resize cursor and unselectable text.
  useEffect(() => () => { endRef.current(true); }, []);

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    // Left button only: the map's region select is a RIGHT-drag, and a right-press that lands a
    // few pixels onto the handle should still start it.
    if (e.button !== 0) return;
    const el = ref.current, ws = workspace(), chat = chatPane();
    if (!el || !ws || !chat) return;
    // Deliberately NOT preventDefault: that suppresses the compatibility mouse events, and
    // dblclick is synthesised from those — double-click-to-reset would silently stop working.
    // Text selection is suppressed by .app.resizing instead.
    try { el.setPointerCapture(e.pointerId); } catch { /* */ }
    el.focus();                           // so Escape reaches us mid-drag in every browser
    next.current = chat.offsetWidth;
    drag.current = { id: e.pointerId, x0: e.clientX, w0: chat.offsetWidth, wsW: ws.clientWidth, handle: el.offsetWidth };
    setDragging(true);
    onResizeStart();
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    // A mouseup delivered outside the window — devtools, a second screen, the menu bar — never
    // reaches us. If the button is already up, the drag is over.
    if (e.buttons === 0) { end(true); return; }
    // The chat is docked right, so moving the pointer LEFT widens it.
    next.current = clampW(d.w0 - (e.clientX - d.x0), d.wsW, d.handle);
    if (raf.current) return;              // one write per frame, whatever rate the pointer runs at
    raf.current = requestAnimationFrame(() => {
      raf.current = 0;
      const cur = drag.current;
      if (!cur) return;
      setWidth(next.current);
      publish(next.current, cur.wsW, cur.handle);
    });
  };

  /** Keyboard and double-click both land here: apply and commit in one go. */
  const jump = (chatW: number) => {
    const ws = workspace(), el = ref.current;
    if (!ws || !el) return;
    const w = clampW(chatW, ws.clientWidth, el.offsetWidth);
    setWidth(w);
    publish(w, ws.clientWidth, el.offsetWidth);
    onResizeEnd(w);
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const ws = workspace(), el = ref.current, chat = chatPane();
    if (!ws || !el || !chat) return;
    const cur = chat.offsetWidth;
    const step = e.shiftKey ? STEP_FAST_PX : STEP_PX;
    const usable = ws.clientWidth - el.offsetWidth;
    switch (e.key) {
      // Left widens the chat because the chat is the RIGHT-hand pane: the keys follow the seam,
      // not the announced value.
      case 'ArrowLeft':  jump(cur + step); break;
      case 'ArrowRight': jump(cur - step); break;
      case 'Home':       jump(usable - mapMin); break;  // seam hard left: map at its minimum
      case 'End':        jump(chatMin); break;          // seam hard right: chat at its minimum
      // A DEVIATION from the ARIA window-splitter pattern, which puts collapse on Enter.
      // Collapsing the map here would mean a width of 0, which UNMOUNTS it and takes the camera
      // with it; hiding the map already has a labelled button in the composer, one Tab away.
      // Reset is the thing keyboard users otherwise have no route to at all.
      case 'Enter':      jump(defaultWidth); break;
      // Only while a drag is in flight, so Escape stays available to everything else.
      case 'Escape':     if (!drag.current) return; end(false); break;
      default: return;
    }
    e.preventDefault();
  };

  return (
    <div ref={ref} className="splitter"
      role="separator" tabIndex={0} aria-orientation="vertical"
      aria-controls="mapwrap"
      aria-label="Resize the map and chat panes"
      aria-valuemin={val.min} aria-valuemax={val.max} aria-valuenow={val.now}
      aria-valuetext={`Map ${val.now}%, chat ${100 - val.now}%`}
      title="Drag to resize · double-click to reset · ←/→ to nudge"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={() => end(true)}
      onPointerCancel={() => end(true)}
      onLostPointerCapture={() => end(true)}
      onDoubleClick={() => jump(defaultWidth)}
      onDragStart={(e) => e.preventDefault()}
      onKeyDown={onKeyDown}
    />
  );
}
