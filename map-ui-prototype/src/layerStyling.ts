// What the agent chose to show of a layer's features: point size, a highlight, labels, line
// width and colour (add_map_layer, Stage 80). Pure functions, so they are checked without deck.gl
// (layerStyling.check.ts); toDeckLayer draws with them.
//
// Live, the Ridgecrest aftershocks answer said "sized by magnitude, the mainshock is the largest
// point" over 1,054 equal circles: the size was a constant here and nothing could single a
// feature out. The server validates every option against the data and states what it draws;
// this side draws exactly that, and drops anything malformed rather than guessing at it.
import type { Highlight, Labels, RGBA, SizeBy } from './contracts';

export interface LayerStyling {
  sizeBy?: SizeBy;
  pointSize?: number;
  highlight?: Highlight;
  labels?: Labels;
  lineWidth?: number;
  lineColor?: RGBA;
}

const num = (v: any): number | undefined => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() ? Number(v) : NaN;
  return Number.isFinite(n) ? n : undefined;
};

const rgba = (c: any): RGBA | undefined => {
  if (!Array.isArray(c) || c.length < 3 || !c.slice(0, 4).every((v: any) => Number.isFinite(Number(v)))) return undefined;
  return [Number(c[0]), Number(c[1]), Number(c[2]), c.length > 3 ? Number(c[3]) : 255];
};

const OPS = new Set(['eq', 'in', 'gte', 'lte', 'gt', 'lt']);

/** The styling fields of a `map_layer` event, snake_case as the agent sends them. Each is kept
 *  only when well formed; without any, the layer draws exactly as before these options existed. */
export function stylingFromWire(layer: any): LayerStyling {
  const out: LayerStyling = {};
  const sb = layer?.size_by ?? layer?.sizeBy;
  if (sb && typeof sb.column === 'string' && Array.isArray(sb.range) && sb.range.length === 2) {
    const min = num(sb.min), max = num(sb.max), lo = num(sb.range[0]), hi = num(sb.range[1]);
    if (min !== undefined && max !== undefined && lo !== undefined && hi !== undefined && hi > lo && lo > 0) {
      out.sizeBy = { column: sb.column, min, max, range: [lo, hi] };
    }
  }
  const ps = num(layer?.point_size ?? layer?.pointSize);
  if (ps !== undefined && ps > 0) out.pointSize = ps;
  const hl = layer?.highlight;
  if (hl && typeof hl.column === 'string' && OPS.has(hl.op) && 'value' in hl) {
    out.highlight = {
      column: hl.column, op: hl.op, value: hl.value, numeric: hl.numeric === true,
      label: typeof hl.label === 'string' && hl.label ? hl.label : hl.column,
      color: rgba(hl.color) ?? HIGHLIGHT,
      ...(Number.isInteger(hl.count) ? { count: hl.count } : {}),
    };
  }
  const lb = layer?.labels;
  if (lb && Array.isArray(lb.items)) {
    const items = lb.items
      .filter((it: any) => it && Array.isArray(it.position) && it.position.length === 2
        && num(it.position[0]) !== undefined && num(it.position[1]) !== undefined && it.text != null)
      .slice(0, 50)
      .map((it: any) => ({ position: [Number(it.position[0]), Number(it.position[1])] as [number, number],
                           text: String(it.text) }));
    if (items.length) out.labels = { column: String(lb.column ?? ''), scope: lb.scope === 'highlighted' ? 'highlighted' : 'all', items };
  }
  const lw = num(layer?.line_width ?? layer?.lineWidth);
  if (lw !== undefined && lw > 0) out.lineWidth = lw;
  const lc = rgba(layer?.line_color ?? layer?.lineColor);
  if (lc) out.lineColor = lc;
  return out;
}

// The server's default (layer_styling.HIGHLIGHT_COLOUR), for a highlight that came without one.
export const HIGHLIGHT: RGBA = [255, 193, 7, 255];

/** Whether a feature's properties match the highlight — the same rule as the server's
 *  layer_styling.highlight_mask, which counted the matches the tool result reports. */
export function highlightTest(h: Highlight): (props: Record<string, any> | null | undefined) => boolean {
  const values = h.op === 'in' ? (Array.isArray(h.value) ? h.value : [h.value]) : [h.value];
  if (h.numeric) {
    const nums = values.map(Number);
    const v = nums[0];
    return (p) => {
      const x = num((p || {})[h.column]);
      if (x === undefined) return false;
      switch (h.op) {
        case 'eq': return x === v;
        case 'in': return nums.includes(x);
        case 'gte': return x >= v;
        case 'lte': return x <= v;
        case 'gt': return x > v;
        case 'lt': return x < v;
      }
      return false;
    };
  }
  const texts = values.map((v) => String(v));
  return (p) => {
    const x = (p || {})[h.column];
    if (x === null || x === undefined) return false;
    return texts.includes(String(x));
  };
}

/** A point's radius in pixels from its value: lo + (hi - lo) * sqrt(t), t its place in the
 *  column's range. A feature with no value is drawn at the smallest size. */
export function sizeScale(s: SizeBy): (props: Record<string, any> | null | undefined) => number {
  const [lo, hi] = s.range;
  const span = s.max - s.min || 1;
  return (p) => {
    const v = num((p || {})[s.column]);
    if (v === undefined) return lo;
    const t = Math.min(1, Math.max(0, (v - s.min) / span));
    return lo + (hi - lo) * Math.sqrt(t);
  };
}
