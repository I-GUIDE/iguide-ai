// A LOCAL, DETERMINISTIC stand-in for the I-GUIDE agent -- pure keyword parsing,
// NO LLM and NO connection to the main agentic system. Its whole job is to be
// the single swap point: replace parseIntent()+the executor with either the real
// I-GUIDE agent endpoint or an LLM tool-router. See README "Where the real
// system plugs in".
import { OVERPASS_PRESETS, type OverpassPreset } from './overpass';

export type Intent =
  | { kind: 'kb'; query: string; useRegion: boolean }
  | { kind: 'overpass'; preset: OverpassPreset; useRegion: boolean }
  | { kind: 'relate'; preset: OverpassPreset; relation: 'intersects' | 'within'; targetHint: string }
  | { kind: 'analysis'; op: 'buffer' | 'clip' | 'hull' | 'heatmap'; km: number; targetHint: string }
  | { kind: 'clear' }
  | { kind: 'help' }
  | { kind: 'clarify'; text: string };

const REGION_WORDS = /\b(here|this region|in the region|in this area|this area|nearby|around here|in view|on screen|on-screen|visible)\b/;
const RELATION_WORDS = /\b(interact|interacts|intersect|intersects|intersecting|cross|crosses|crossing|touch|touches|overlap|overlaps|overlapping|pass through|passing through|through|within|inside|contained)\b/;
const TARGET_WORDS = /\b(upload|uploaded|geojson|my data|my layer|the layer|study area|the polygon|the shape|the boundary|it|them|these)\b/;

const OVERPASS_SYNONYMS: Record<string, string[]> = {
  cafe: ['cafe', 'cafes', 'café', 'cafés', 'coffee'],
  restaurant: ['restaurant', 'restaurants', 'dining', 'eatery'],
  school: ['school', 'schools'],
  hospital: ['hospital', 'hospitals', 'clinic', 'clinics'],
  park: ['park', 'parks', 'green space'],
  supermarket: ['supermarket', 'supermarkets', 'grocery', 'groceries', 'grocer'],
  river: ['river', 'rivers', 'waterway', 'waterways'],
  stream: ['stream', 'streams', 'creek', 'creeks'],
  water: ['water body', 'water bodies', 'lake', 'lakes', 'pond', 'ponds', 'reservoir'],
  road: ['road', 'roads', 'highway', 'highways', 'street', 'streets'],
};

function findPreset(text: string): OverpassPreset | null {
  for (const p of OVERPASS_PRESETS) {
    const syns = OVERPASS_SYNONYMS[p.key] ?? [p.key];
    if (syns.some((s) => text.includes(s))) return p;
  }
  return null;
}

// Explicit knowledge-base cues -- KB is NO LONGER the catch-all fallback.
const KB_CUES = /\b(find|search|dataset|datasets|notebook|notebooks|publication|publications|paper|papers|knowledge base|\bkb\b|data on|data about|data for|datasets? about|related datasets?)\b/;

export function parseIntent(text: string): Intent {
  const t = text.toLowerCase().trim();

  if (/\b(clear|reset|start over|wipe)\b/.test(t)) return { kind: 'clear' };
  if (/^\s*(help|what can you|how do i|examples?)\b/.test(t)) return { kind: 'help' };

  const preset = findPreset(t);

  // Compositional: "<osm entity> that interact with <the uploaded layer>"
  if (preset && RELATION_WORDS.test(t) && TARGET_WORDS.test(t)) {
    const relation = /\b(within|inside|contained)\b/.test(t) ? 'within' : 'intersects';
    return { kind: 'relate', preset, relation, targetHint: t };
  }

  // Analysis verbs (act on an existing layer).
  if (/\bbuffer\b/.test(t)) {
    const m = t.match(/(\d+(?:\.\d+)?)\s*(km|kilomet|mi|mile)/);
    const km = m ? parseFloat(m[1]) * (/mi|mile/.test(m[2]) ? 1.60934 : 1) : 5;
    return { kind: 'analysis', op: 'buffer', km, targetHint: t };
  }
  if (/\b(heatmap|heat map|density|hotspot|hot spot)\b/.test(t))
    return { kind: 'analysis', op: 'heatmap', km: 0, targetHint: t };
  if (/\b(convex hull|hull|extent|footprint)\b/.test(t))
    return { kind: 'analysis', op: 'hull', km: 0, targetHint: t };
  if (/\b(clip|restrict to|only in|keep in)\b/.test(t) && !KB_CUES.test(t))
    return { kind: 'analysis', op: 'clip', km: 0, targetHint: t };

  // Plain OSM entity request ("show cafés here", "rivers in view").
  if (preset) return { kind: 'overpass', preset, useRegion: REGION_WORDS.test(t) || true };

  // Knowledge base only on explicit cues.
  if (KB_CUES.test(t)) {
    const query = text
      .replace(/\b(find|search|show me|show|get|please|the|any|all|datasets?|dataset|data|notebooks?|publications?|papers?|about|on|for|related to|knowledge base|kb|in the region|here|nearby|near me)\b/gi, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    return { kind: 'kb', query: query || text, useRegion: REGION_WORDS.test(t) };
  }

  // Ambiguous -> ask, DON'T silently run a KB search.
  return { kind: 'clarify', text };
}

// Starter prompts for the LIVE agent. The first two need nothing set up; the rest name
// their prerequisite (an upload, a drawn region) so it is obvious what to do first.
// These replace the mock-era examples ("Show cafés here", "Buffer the cafés by 2 km"),
// which exercised the offline demo rather than the real toolkit.
export const SUGGESTIONS = [
  'Find flood risk datasets on I-GUIDE',
  'Show hospitals near Chicago on the map',
  'Aggregate my uploaded points into a 1 km grid',
  'Embed the selected region with the GSE model',
];
