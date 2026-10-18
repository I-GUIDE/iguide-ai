/** Loading one `map_layer` event's GeoJSON, with every way it can fail named.
 *
 * Kept out of App.tsx so `npm run check:layers` can run it. A layer that fails to load used to
 * vanish without a word: on 2026-10-17 (round r59) the route file held bare NaN, JSON.parse threw
 * "Unexpected token 'N'", an empty `catch {}` swallowed it, and the answer beside the empty map
 * said the route was on it. Each failure now comes back as a sentence the trace and the answer
 * card can show. */
import type { FeatureCollection } from 'geojson';

export type LayerLoad =
  | { fc: FeatureCollection }
  /** Not on the map, and why, in words a reader can act on. */
  | { problem: string }
  /** Stop was pressed: the reader knows why the layer is missing, so nothing is said. */
  | { aborted: true };

export async function loadLayerData(
  inline: unknown,
  url: string | undefined,
  fetchIt: (url: string) => Promise<Response>,
): Promise<LayerLoad> {
  let data: unknown = inline;
  if (!data && url) {
    let res: Response;
    try {
      res = await fetchIt(url);
    } catch (e: any) {
      if (e?.name === 'AbortError') return { aborted: true };
      return { problem: `the download failed: ${e?.message || e}` };
    }
    if (!res.ok) {
      // A refused download used to vanish here without a word, beside an answer saying the
      // layer was on the map.
      const why = await res.json().then((j: any) => j?.reason || '').catch(() => '');
      return { problem: `HTTP ${res.status}${why ? `, ${why}` : ''}` };
    }
    let text: string;
    try {
      text = await res.text();
    } catch (e: any) {
      if (e?.name === 'AbortError') return { aborted: true };
      return { problem: `the download failed: ${e?.message || e}` };
    }
    try {
      data = JSON.parse(text);
    } catch (e: any) {
      return { problem: `its file is not valid JSON: ${e?.message || e}` };
    }
  }
  if (!data) return { problem: 'the event carried no data and no file to fetch' };
  const fc = data as FeatureCollection;
  if (!Array.isArray(fc.features)) return { problem: 'its file is not a GeoJSON FeatureCollection' };
  if (!fc.features.length) return { problem: 'its file holds no features' };
  return { fc };
}

/** The trace line and answer-card note for a layer that did not draw. */
export function undrawnLine(label: string, problem: string): string {
  return `map: ${label} could not be loaded (${problem}), so it is not on the map`;
}
