import type { ReplayMeta } from '../replay';

/** Replay builds only: `?replay=<key>` runs a recorded turn through the real client instead of
 *  a live stream (see src/replay.ts). Written as a direct comparison so a production build folds
 *  it to `null` — the same reason uiVariant.ts gives. */
export const REPLAY_KEY: string | null =
  (import.meta.env.DEV || import.meta.env.VITE_REPLAY === '1')
    ? new URLSearchParams(location.search).get('replay')
    : null;

export interface ReplayVerdict { drawn: number; total: number; distinct: number; lines: string[] }

/** What the screenshot has to say that the page alone would not: that this is a recording, and
 *  which of its map layers did not make it onto the map, and why. A layer that silently fails to
 *  draw would otherwise look exactly like a turn that never placed one. */
export function ReplayStrip({ meta, verdict, error }: {
  meta: ReplayMeta | null; verdict: ReplayVerdict | null; error: string | null;
}) {
  if (error) return <div className="replay-strip bad"><b>REPLAY FAILED</b> · {error}</div>;
  if (!meta) return <div className="replay-strip">REPLAY · loading the recording…</div>;
  const bad = (!!verdict && verdict.lines.length > 0) || meta.answerFrom === 'none';
  return (
    <div className={`replay-strip${bad ? ' bad' : ''}`}>
      <div>
        <b>REPLAY</b> · {meta.title} · recorded stream, {meta.events} events
        {meta.badLines ? ` (${meta.badLines} unreadable lines skipped)` : ''}
        {' · '}
        {verdict
          ? `map layers: ${verdict.drawn} of ${verdict.total} drawn`
            + (verdict.distinct < verdict.total
              ? ` (${verdict.distinct} on the list: a repeated id replaces the earlier layer)` : '')
          : 'replaying…'}
        {meta.answerFrom === 'record' && ' · answer taken from the record (the recording has no answer event)'}
        {meta.answerFrom === 'none' && ' · NO ANSWER: neither the recording nor the record has one'}
        {meta.rewrittenHosts?.length
          ? ` · file urls on ${meta.rewrittenHosts.join(', ')} served from the capture` : ''}
        {meta.geojsonParsed
          ? ` · ${meta.geojsonParsed} inline layer${meta.geojsonParsed === 1 ? '' : 's'} stored as text, parsed back` : ''}
        {meta.queryFrom === 'route_trace' && ' · query taken from the route trace (no harness record)'}
        {meta.queryFrom === 'none' && ' · the query was not recorded'}
      </div>
      {verdict?.lines.map((l, i) => <div key={i} className="replay-line">{l}</div>)}
    </div>
  );
}
