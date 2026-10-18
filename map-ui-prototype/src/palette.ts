// The map's own colours for classes that carry no colour of their own. A tool sends class names,
// and a colour only where it MEANS something (Gi* hot spots red, LISA Low-Low blue) or the user
// asked for one (add_map_layer's class_colors); everything else is coloured here, so the look of
// every categorical map is decided in one place. Same twelve hues, in the same order, that the
// agent used to send with every layer, so maps look as they did.
import type { Legend, RGBA } from './contracts';

export const CATEGORY_PALETTE: RGBA[] = [
  [215, 25, 28, 200], [44, 123, 182, 200], [253, 174, 97, 200], [171, 217, 233, 200],
  [26, 150, 65, 200], [123, 50, 148, 200], [255, 217, 47, 200], [166, 97, 26, 200],
  [230, 97, 158, 200], [102, 194, 165, 200], [140, 140, 140, 200], [8, 81, 156, 200],
];

const same = (a: RGBA, b: RGBA) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];

/** Every legend entry with a colour: its own where it came with one, otherwise the next palette
 *  hue that no entry was given explicitly (so "Lozano in red" does not leave a second red class).
 *  Order is the legend's, which is the order the classes were sent in. */
export function resolveLegend(legend: Legend[] | undefined): (Legend & { color: RGBA })[] {
  if (!legend?.length) return [];
  const taken = legend.filter((e) => e.color).map((e) => e.color as RGBA);
  const free = CATEGORY_PALETTE.filter((c) => !taken.some((t) => same(t, c)));
  let next = 0;
  return legend.map((e) => {
    if (e.color) return { ...e, color: e.color };
    const color = free.length ? free[next++ % free.length] : CATEGORY_PALETTE[next++ % CATEGORY_PALETTE.length];
    return { ...e, color };
  });
}

/** A layer's single colour as a fill and an outline: the fill keeps the basemap readable. */
export function fillAndLine(c: RGBA): { fill: RGBA; line: RGBA } {
  return { fill: [c[0], c[1], c[2], Math.min(c[3], 190)], line: [c[0], c[1], c[2], 255] };
}
