import { useCallback, useEffect, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import type { Feature, FeatureCollection, Polygon, Geometry } from 'geojson';
import { AgentMap } from './components/AgentMap';
import { ChatPanel, type ChatMessage, type Mode, type AgentCfg } from './components/ChatPanel';
import { TopNav } from './components/TopNav';
import { LeftPanel, type SelectedFeature } from './components/LeftPanel';
import { Splitter } from './components/Splitter';
import type { LayerArtifact } from './contracts';
import { parseIntent } from './agentBrain';
import { searchKb, kbHitsToFeatureCollections, type KbHit } from './mockKb';
import { queryOverpass } from './overpass';
import { bufferFC, clipToRegion, convexHull, areaKm2, stats, selectRelated, layerBBox } from './analysis';
import { bboxToFC } from './mapFit';
import {
  streamChat, uploadFiles, absoluteUrl, extractFeatures, newThreadId, fetchModels, fetchUiConfig,
  type AgentConfig, type FileRecord, type ModelCatalogue, type TraceLine } from './agentClient';
import { AuthError, authMessage } from './auth';
import { fetchWhoAmI, listConversations, putConversation, getConversation,
  type WhoAmI } from './agentClient';
import { renderMarkdown } from './markdown';
import type { AppTab } from './uiVariant';
import {
  deleteSession, listSessions, loadSession, newSessionId, saveSession, titleFor,
  toStoredLayer, type SessionSummary, type StoredSession,
} from './sessionStore';

const DEFAULT_CFG: AgentCfg = {
  endpoint: '/agent/chat/stream',       // proxied to the deployed agent by vite (see vite.config)
  uploadEndpoint: '/agent/files/upload',
  apiKey: '',
};

// Retrieval tools that do NOT touch geography — used when the "Spatial tools" toggle
// is off so a pure-chat turn stays fast and never opens the map.
const CHAT_ONLY_METHODS = ['keyword_search', 'semantic_search', 'neo4j_search', 'web_search', 'agent_kb_search', 'get_kb_block'];

// The chat pane's width belongs to the user now. These are mirrored in styles.css as
// --chat-min / --map-min so the stylesheet can re-clamp a restored width on its own, live, as
// the window changes — change them in BOTH places.
const CHAT_W_DEFAULT = 460;   // the width the remote-sensing row was tuned against
const CHAT_W_MIN = 380;       // below this the composer's three 42px circles crowd the textarea out
const MAP_W_MIN = 360;        // .leftpanel floats over the map and needs 312 (288 + gutters)

// The first thing anyone reads, and it has to be actionable by the person reading it. The
// default points at the settings gear; DEMO_MODE hides that gear, so pointing at it there tells
// a visitor to click something that is not on their screen. The demo greeting says what is
// already true instead — spatial tools are on, and forced on below, since nothing can turn them
// off once the gear is gone.
const GREETING = "Hi — I'm the I-GUIDE agent. Ask me anything. Turn on Spatial tools (⚙) to search geodata; the map opens on its own when I return geometry, or hit Map — then right-click or right-drag on it to select a region.";
const GREETING_DEMO = "Hi — I'm the I-GUIDE agent. Ask me anything — spatial tools are on, so I can search geodata and open datasets. The map appears on its own when I return geometry, or hit Map — then right-click or right-drag on it to select a region.";

function loadCfg(): { mode: Mode; cfg: AgentCfg; spatial: boolean; chatW: number } {
  try {
    const raw = localStorage.getItem('iguide-map-ui');
    if (raw) { const j = JSON.parse(raw); return { mode: j.mode || 'live', cfg: { ...DEFAULT_CFG, ...j.cfg }, spatial: j.spatial !== false, chatW: readChatW(j.chatW) }; }
  } catch { /* */ }
  return { mode: 'live', cfg: DEFAULT_CFG, spatial: true, chatW: CHAT_W_DEFAULT };
}
// A stored width is not to be trusted: an older blob has no field at all, and JSON round-trips
// NaN and Infinity to null. Only the FLOOR is enforced here — the ceiling depends on the window,
// and styles.css applies that live in clamp(), so a width saved on a wide monitor cannot starve
// the map on a laptop and the user's choice comes back when the room returns.
function readChatW(v: unknown): number {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.max(n, CHAT_W_MIN) : CHAT_W_DEFAULT;
}

function bboxPolygon(a: [number, number], b: [number, number]): Polygon {
  const minLon = Math.min(a[0], b[0]), maxLon = Math.max(a[0], b[0]);
  const minLat = Math.min(a[1], b[1]), maxLat = Math.max(a[1], b[1]);
  return { type: 'Polygon', coordinates: [[[minLon, minLat], [maxLon, minLat], [maxLon, maxLat], [minLon, maxLat], [minLon, minLat]]] };
}
function viewportPolygon(): Polygon | null {
  const m = (window as any).__map; if (!m) return null;
  // A handle can outlive its map — the pane is taken away by more than the Hide button now —
  // and unlike applyFit this result is spliced straight into the prompt we send.
  try {
    const b = m.getBounds();
    return bboxPolygon([b.getWest(), b.getSouth()], [b.getEast(), b.getNorth()]);
  } catch { return null; }
}
function polygonBBox(poly: Polygon): [number, number, number, number] {
  const ring = poly.coordinates[0]; const lons = ring.map((c) => c[0]); const lats = ring.map((c) => c[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

export default function App() {
  // loadCfg reads localStorage and parses JSON. As a bare call in the body it did that on EVERY
  // render, for values only the initial useState arguments ever read; lazily, it runs once.
  const [init] = useState(loadCfg);
  const [layers, setLayers] = useState<LayerArtifact[]>([]);
  const [drawnRegion, setDrawnRegion] = useState<Polygon | null>(null);
  const [drawPreview, setDrawPreview] = useState<Feature | null>(null);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<Mode>(init.mode);
  const [cfg, setCfg] = useState<AgentCfg>(init.cfg);
  const [chatW, setChatW] = useState<number>(init.chatW);
  // The drag class on .app — two renders per gesture, not one per frame.
  const [resizing, setResizing] = useState(false);
  // refitAfterResize is memoized with [] and reads its inputs from refs, so the drag flag is one too.
  const resizingRef = useRef(false);
  const [spatial, setSpatial] = useState<boolean>(init.spatial);
  const [mapVisible, setMapVisible] = useState(false);
  const [tab, setTab] = useState<AppTab>('chat');
  // Where the RS-Embed Demo tab lands you. The demo is about this pair of cities, and a map
  // opened over the whole country asks the visitor to go and find the subject first.
  const CHAMPAIGN_URBANA: [number, number, number, number] = [-88.32, 40.05, -88.14, 40.16];
  const [showSettings, setShowSettings] = useState(false);
  // Which models this agent will accept. Fetched once so the picker offers what is actually
  // served rather than a hardcoded list that drifts; null just means "agent default only".
  const [models, setModels] = useState<ModelCatalogue | null>(null);
  const [selected, setSelected] = useState<SelectedFeature | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'agent', text: GREETING },
  ]);

  const threadRef = useRef<string>(newThreadId());
  // One local history record per conversation. The server already owns the conversational
  // memory (keyed by memoryId); this is the client remembering which conversations exist.
  const sessionIdRef = useRef<string>(newSessionId());
  // EVERY file uploaded in this session. pendingFileIds is cleared once a turn attaches them
  // server-side, but a RESTORED session has to re-attach them all: the analysis tools only
  // load when files are attached, so without this a restored session loses the whole toolkit.
  const sessionFileIds = useRef<string[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const memoryRef = useRef<string | null>(null);
  const pendingFileIds = useRef<string[]>([]);
  const uploadContext = useRef<string>(''); // e.g. 'Uploaded "chicago" covers bbox [...]' — spatial context for the agent
  const layersRef = useRef(layers); layersRef.current = layers;
  const abortRef = useRef<AbortController | null>(null);

  // chatW changes once per gesture (release, key step, reset), never per frame, so it can ride
  // the same single record as everything else. It has to be in BOTH the payload and the deps: in
  // the payload alone it would only ever be flushed by an unrelated settings change, and a
  // SEPARATE effect writing this key would clobber cfg — setItem replaces the whole object.
  useEffect(() => { try { localStorage.setItem('iguide-map-ui', JSON.stringify({ mode, cfg, spatial, chatW })); } catch { /* */ } }, [mode, cfg, spatial, chatW]);

  // Progressive map: auto-reveal ONCE the first time a layer appears, then respect the
  // manual show/hide toggle (don't fight the user). Re-arm after layers are cleared.
  const autoRevealed = useRef(false);
  useEffect(() => {
    if (layers.length && !autoRevealed.current) { autoRevealed.current = true; setMapVisible(true); }
    if (!layers.length) autoRevealed.current = false;
  }, [layers.length]);
  // The instance is destroyed when hidden, so drop the stale handle.
  useEffect(() => { if (!mapVisible) (window as any).__map = undefined; }, [mapVisible]);
  useEffect(() => {
    const el = mapBoxRef.current;
    if (!mapVisible || !el) { setMapBoxReady(false); return; }
    const measure = () => setMapBoxReady(el.clientWidth > 0 && el.clientHeight > 0);
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [mapVisible]);

  const asAgentConfig = useCallback((): AgentConfig => ({ ...cfg }), [cfg]);
  // DEMO_MODE on the server: the API key is not enforced and the connection settings are hidden,
  // so a link can be handed to an audience without also handing them a credential to paste.
  // Asked once per endpoint, and only in live mode — the offline demo has no server to ask.
  // Unreachable or unknown means NOT a demo, which keeps the settings available: a page that
  // hides the key field on a deployment that turns out to need one cannot be recovered from.
  const [demoMode, setDemoMode] = useState(false);
  // Token mode hides the connection settings for a different reason than demo does: there is a
  // credential, it is just not one you paste — the browser holds it as a cookie.
  const [tokenMode, setTokenMode] = useState(false);
  // Who the SERVER says we are. The access cookie is httpOnly, so the page cannot answer this
  // itself. Null means "nobody", which lists nothing rather than everything.
  //
  // The whole answer is kept, not just the id: the id is what scopes stored conversations, but
  // the role, the permitted flag and the reason are what the account badge needs to say where
  // someone stands BEFORE they spend a question finding out.
  const [me, setMe] = useState<WhoAmI | null>(null);
  const [viewer, setViewer] = useState<string | null>(null);
  const viewerRef = useRef<string | null>(null);
  useEffect(() => { viewerRef.current = viewer; }, [viewer]);
  useEffect(() => {
    if (mode !== 'live') { setDemoMode(false); setTokenMode(false); return; }
    let live = true;
    void fetchUiConfig(asAgentConfig()).then((c) => {
      if (!live) return;
      const demo = !!c?.demo_mode;
      setDemoMode(demo);
      setTokenMode(c?.mode === 'token');
      if (c?.mode === 'token') {
        void fetchWhoAmI(asAgentConfig()).then((who) => {
          if (!live) return;
          setMe(who);
          setViewer(who?.user?.id ?? null);
        });
      } else {
        // Dev and demo identify nobody, so there is no badge and no owner to scope by.
        setMe(null);
        setViewer(null);
      }
      if (!demo) return;
      // The config arrives after the first paint, so the greeting is already on screen and has
      // to be replaced — but only while it is still the whole conversation. A restored session,
      // or a visitor who typed before the answer came back, keeps what it has.
      setMessages((m) => (m.length === 1 && m[0].role === 'agent' && m[0].text === GREETING
        ? [{ ...m[0], text: GREETING_DEMO }] : m));
      // Nothing can turn these back on with the gear hidden, and a returning visitor who once
      // switched them off would otherwise be stuck with a crippled demo and no way out of it.
      setSpatial(true);
    });
    return () => { live = false; };
  }, [mode, cfg.endpoint]);
  useEffect(() => {
    if (mode !== 'live') return;
    let live = true;
    void fetchModels(asAgentConfig()).then((m) => { if (live) setModels(m); });
    return () => { live = false; };
  }, [mode, cfg.endpoint, cfg.apiKey]);   // re-ask when the target or credential changes
  const resolveUrl = useCallback((u: string) => absoluteUrl(u, asAgentConfig()), [asAgentConfig]);

  const pushMsg = useCallback((m: ChatMessage) => setMessages((prev) => [...prev, m]), []);
  const putLayer = useCallback((a: LayerArtifact) => {
    setLayers((prev) => { const i = prev.findIndex((l) => l.id === a.id); if (i === -1) return [...prev, a]; const n = prev.slice(); n[i] = a; return n; });
  }, []);
  // A layer often arrives before MapLibre has finished loading (a heatmap can be delivered
  // within a second of the map mounting). fitBounds on a not-yet-loaded map is silently
  // dropped, which left the user looking at the whole US with their city as a single dot —
  // so remember the request and apply it the moment the map is ready.
  const pendingFit = useRef<FeatureCollection | null>(null);
  // The last fit we asked for, so it can be re-applied once the container settles (below).
  const lastFit = useRef<{ fc: FeatureCollection; at: number } | null>(null);
  const applyFit = (m: any, fc: FeatureCollection, duration: number) => {
    try {
      const [w, s, e, n] = layerBBox(fc);
      m.fitBounds([[w, s], [e, n]], { padding: 80, duration, maxZoom: 14 });
    } catch { /* degenerate bbox — leave the view alone */ }
  };
  const fitView = useCallback((fc: FeatureCollection) => {
    const m = (window as any).__map;
    if (!fc.features.length) return;
    lastFit.current = { fc, at: Date.now() };
    if (!m) { pendingFit.current = fc; return; }
    applyFit(m, fc, 800);
  }, []);
  // The map instance is published as soon as it EXISTS (deliberately: waiting for `load`
  // strands everything when a basemap tile source fails). So on a first-turn reveal fitBounds
  // can run while the pane is still expanding — it then sizes the view for a container a
  // fraction of the final width and lands ~3 zoom levels too deep, which turned a 31,977-point
  // density surface into one pane-filling blob with the basemap scrolled out of frame.
  // Re-apply the fit when the container resizes, but only right after we asked for it, so a
  // view the user has since panned or zoomed is never yanked back.
  const refitAfterResize = useCallback(() => {
    // A splitter drag resizes the map on every frame. Re-framing per frame makes the map fight
    // the pointer for the 2.5s after any fit — which is exactly when a user reaches for the
    // seam, a layer having just landed. Hold off, and refit once on release instead.
    if (resizingRef.current) return;
    const m = (window as any).__map;
    const last = lastFit.current;
    if (!m || !last || Date.now() - last.at > 2500) return;
    applyFit(m, last.fc, 0);
  }, []);

  // --- local chat history: snapshot, restore, continue ------------------------------
  // Saved after each turn rather than on every keystroke: a turn is the unit a user would
  // expect to come back to, and it keeps writes off the streaming path.
  const layersRef2 = useRef(layers); layersRef2.current = layers;
  const messagesRef = useRef(messages); messagesRef.current = messages;
  /** Where the history list comes from.
   *
   *  In token mode the SERVER owns it: conversations belong to the user, follow them to another
   *  browser, and — the reason this changed — do not sit in IndexedDB waiting for the next
   *  person to open the page. IndexedDB stays the source for dev and demo, which identify
   *  nobody, and stays a local cache in token mode so a restore does not wait on a round trip.
   *
   *  A failed fetch returns null and is NOT treated as "no conversations": showing an empty
   *  history because the server was briefly unreachable reads as data loss. */
  const refreshSessions = useCallback(async () => {
    if (tokenMode && viewerRef.current) {
      const remote = await listConversations(asAgentConfig());
      if (remote) {
        setSessions(remote.map((c) => ({
          id: c.memoryId,
          title: c.conversationName || 'Untitled conversation',
          createdAt: Date.parse(c.createdAt || '') || 0,
          updatedAt: Date.parse(c.updatedAt || '') || 0,
          threadId: c.threadId || '',
          messageCount: c.messageCount ?? 0,
          layerCount: c.layerCount ?? 0,
          fileCount: c.fileCount ?? 0,
        })));
        return;
      }
      return;                       // could not ask; keep whatever is on screen
    }
    setSessions(await listSessions(viewerRef.current));
  }, [tokenMode, asAgentConfig]);

  const snapshotSession = useCallback(() => {
    const msgs = messagesRef.current;
    if (!msgs.some((m) => m.role === 'user')) return;   // nothing worth listing yet
    const now = Date.now();
    const rec: StoredSession = {
      id: sessionIdRef.current,
      title: titleFor(msgs),
      createdAt: now, updatedAt: now,
      threadId: threadRef.current,
      memoryId: memoryRef.current,
      messages: msgs,
      layers: layersRef2.current.map(toStoredLayer),
      fileIds: [...new Set(sessionFileIds.current)],
      region: drawnRegion ?? undefined,
      model: cfg.model, provider: cfg.provider,
    };
    rec.ownerId = viewerRef.current;
    void loadSession(rec.id, viewerRef.current).then((prev) => {
      if (prev) rec.createdAt = prev.createdAt;          // keep the original start time
      return saveSession(rec);
    }).then(() => {
      // Server-side too, keyed by the agent's own memoryId so the record and the conversation
      // it continues are the same thing. Local stays a cache; this is the copy that survives a
      // new browser, and the one another device will read.
      if (tokenMode && viewerRef.current && memoryRef.current) {
        return putConversation(asAgentConfig(), memoryRef.current, rec).then(() => undefined);
      }
      return undefined;
    }).then(() => refreshSessions());
  }, [drawnRegion, cfg.model, cfg.provider, tokenMode, asAgentConfig, refreshSessions]);

  const restoreSession = useCallback(async (id: string) => {
    // Local first — it is a cache, and a hit avoids a round trip. In token mode fall back to
    // the server, which is where a conversation opened on another browser actually lives.
    let rec = await loadSession(id, viewerRef.current);
    if (!rec && tokenMode && viewerRef.current) rec = await getConversation(asAgentConfig(), id);
    if (!rec) return;
    setShowHistory(false);
    // Identity first: the next turn must continue the SAME agent conversation, and must
    // re-attach the files or the analysis tools will not even load.
    sessionIdRef.current = rec.id;
    threadRef.current = rec.threadId;
    memoryRef.current = rec.memoryId ?? null;
    sessionFileIds.current = [...(rec.fileIds || [])];
    pendingFileIds.current = [...(rec.fileIds || [])];
    setMessages(rec.messages || []);
    setLayers([]);
    autoRevealed.current = false;
    // Geometry was never stored — re-fetch each layer from the file store, which keeps its
    // files indefinitely. A layer with no sourceUrl (a local upload preview, a mock preview)
    // cannot come back; it is skipped rather than restored empty.
    const restored: LayerArtifact[] = [];
    for (const l of rec.layers || []) {
      if (l.kind === 'raster') { restored.push(l); continue; }
      const url = (l as any).sourceUrl;
      // A layer delivered inline (overpass_search, spatial_search) has no url behind it, so its
      // stored geometry IS the copy — use it directly instead of dropping the layer.
      const inline = (l as any).data;
      if (!url) {
        if (inline?.features?.length) restored.push(l as LayerArtifact);
        continue;
      }
      try {
        const res = await fetch(resolveUrl(url));
        if (!res.ok) continue;
        const fc = await res.json();
        if (!fc || !Array.isArray(fc.features) || !fc.features.length) continue;
        restored.push({ ...(l as any), data: fc } as LayerArtifact);
      } catch { /* the file is gone — leave the layer out rather than showing an empty one */ }
    }
    setLayers(restored);
    if (restored.length) {
      setMapVisible(true);
      const first = restored.find((l) => l.kind === 'geojson') as any;
      if (first?.data) fitView(first.data);
    }
    const missing = (rec.layers || []).length - restored.length;
    pushMsg({ role: 'agent', text:
      `Continuing "${rec.title}" — ${(rec.messages || []).length} messages, ` +
      `${restored.length} layer(s) restored${missing > 0 ? `, ${missing} no longer available` : ''}` +
      `${(rec.fileIds || []).length ? `, ${(rec.fileIds || []).length} file(s) re-attached` : ''}.` });
  }, [resolveUrl, fitView, pushMsg]);

  const startNewSession = useCallback(() => {
    sessionIdRef.current = newSessionId();
    threadRef.current = newThreadId();
    memoryRef.current = null;
    sessionFileIds.current = [];
    pendingFileIds.current = [];
    uploadContext.current = '';
    setLayers([]);
    setDrawnRegion(null);
    autoRevealed.current = false;
    setMapVisible(false);
    setShowHistory(false);
    setMessages([{ role: 'agent', text: "New conversation. Ask me anything." }]);
  }, []);

  // `viewer` is in the deps even though refreshSessions reads it from a REF. The ref exists so
  // the callback is not rebuilt on every identity change, but that also meant nothing re-ran
  // the list when identity finally arrived: whoami lands after ui-config, so token mode listed
  // the local cache once and kept showing that count until someone clicked History. Live, that
  // read as "History (27)" for an account with no conversations at all.
  useEffect(() => { void refreshSessions(); }, [refreshSessions, viewer]);

  // --- layer management + feature inspection (left panel) ---
  const toggleLayer = useCallback((id: string) => {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, visible: !(l.visible !== false) } : l)));
  }, []);
  const removeLayerById = useCallback((id: string) => {
    setLayers((prev) => prev.filter((l) => l.id !== id));
  }, []);
  const setLayerOpacity = useCallback((id: string, opacity: number) => {
    setLayers((prev) => prev.map((l) => (l.id === id && l.kind === 'raster' ? { ...l, opacity } : l)));
  }, []);
  const fitLayer = useCallback((id: string) => {
    const l = layersRef.current.find((x) => x.id === id);
    if (!l) return;
    fitView(l.kind === 'raster' ? bboxToFC(l.bounds) : l.data);
  }, [fitView]);
  // Vector artifacts (a .geojson the agent WROTE, e.g. from execute_code) belong on the
  // interactive map, not just in a download list. Fetch and add each one once. Static
  // images (PNG plots) stay as attachments — that is the other visualization route.
  const loadedArtifacts = useRef<Set<string>>(new Set());
  // Set when a turn delivers a layer through the map_layer event. The agent has then
  // said exactly what belongs on the map, so its other .geojson artifacts — the raw
  // conversion it made on the way, or the same file a second time — must NOT be
  // auto-loaded too: three stacked layers of 128,855 points buried the heatmap under
  // a solid mass of circles.
  const mapLayerDelivered = useRef(false);
  // …and the files those layers were drawn FROM, for the whole conversation rather than one
  // turn. mapLayerDelivered resets each turn, so a later turn that places no layer of its own
  // (e.g. "what parameters were used?", answered from memory) ran the artifact fallback over a
  // downloads list that still carried EARLIER turns' files — re-adding the boundary the
  // map_layer event had already drawn, as a second filled layer over the top of it.
  const layerSourceFiles = useRef<Set<string>>(new Set());
  // A map mounted into a zero-sized container (collapsed pane, hidden tab) never finishes
  // initialising — MapLibre does not fire `load`, so there is no instance to resize later and
  // the canvas stays blank. Mount only once the container actually has room.
  const mapBoxRef = useRef<HTMLDivElement | null>(null);
  const [mapBoxReady, setMapBoxReady] = useState(false);
  // Compare files by their file_id, not the URL string: the map_layer descriptor and the
  // download record for the same file differ in absolute-vs-relative form.
  const fileKey = (u?: string | null) => {
    if (!u) return '';
    const m = /\/agent\/files\/([^/]+)\/download/.exec(u);
    return m ? m[1] : u;
  };
  const loadVectorArtifacts = useCallback(async (files: FileRecord[]) => {
    if (!spatial) return;
    for (const f of files) {
      const name = f.filename || f.download_url || '';
      if (!/\.(geo)?json$/i.test(name)) continue;
      const key = f.file_id || f.download_url;
      if (!key || loadedArtifacts.current.has(key)) continue;
      // Already on the map because a map_layer event drew it — possibly several turns ago.
      if (layerSourceFiles.current.has(f.file_id || '') ||
          layerSourceFiles.current.has(fileKey(f.download_url))) continue;
      loadedArtifacts.current.add(key);
      try {
        const res = await fetch(resolveUrl(f.download_url));
        if (!res.ok) continue;
        const j = await res.json();
        const fc: FeatureCollection =
          j?.type === 'FeatureCollection' ? j
          : j?.type === 'Feature' ? { type: 'FeatureCollection', features: [j] }
          : j?.type && j?.coordinates ? { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: j, properties: {} }] }
          : { type: 'FeatureCollection', features: [] };
        if (!fc.features.length) continue;
        const label = (f.filename || 'result').replace(/\.(geo)?json$/i, '');
        // Key the LAYER by filename, not file_id: when the agent regenerates the same
        // output (a retry, or a second pass over the same step) it gets a new file_id but
        // the same name, and the user should see one updated layer rather than two
        // identical ones stacked on the map.
        const layerId = `artifact-${label.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
        // Same rule add_map_layer applies for render:"auto" — a big pile of points is a
        // density surface, not 32,000 overlapping circles. Without this, a turn that wrote
        // its GeoJSON with execute_code (no map_layer event, so we land here) drew a solid
        // orange blob over Chicago and called it a heat map.
        const heat =
          fc.features.length > 2000 &&
          fc.features.every((ft) => /^(Multi)?Point$/.test(ft.geometry?.type || ''));
        putLayer({
          kind: 'geojson', id: layerId, source: 'analysis', label: `${label}`,
          data: fc, fitBounds: true, sourceUrl: f.download_url,
          render: heat ? 'heatmap' : 'geojson',
          style: heat
            ? { opacity: 0.85 }
            : { fill: [201, 138, 26, 110], line: [201, 138, 26, 255], lineWidth: 2, pointRadius: 5 },
        });
        fitView(fc);
      } catch { /* not loadable as GeoJSON — leave it as a download */ }
    }
  }, [spatial, resolveUrl, putLayer, fitView]);

  const onFeatureClick = useCallback((feature: any, layerId: string) => {
    const props = (feature && feature.properties) || {};
    const layer = layersRef.current.find((l) => l.id === layerId);
    const name = props.name || props.title || props.id || props.doc_id || '(feature)';
    setSelected({ name: String(name), layerLabel: layer?.label || layerId, properties: props });
  }, []);

  // Region selection is RIGHT-DRAG on the map (matching the rs-embed demo): press the right
  // button, sweep a box, release. A right-click that barely moves becomes a small box, so a
  // single click still selects somewhere. Left-click stays free for inspecting features.
  const onRegionPreview = useCallback((poly: Polygon | null) => {
    setDrawPreview(poly ? { type: 'Feature', geometry: poly, properties: {} } : null);
  }, []);
  const onRegionDrawn = useCallback((poly: Polygon, viaClick: boolean) => {
    setDrawnRegion(poly);
    const km2 = areaKm2(poly).toLocaleString(undefined, { maximumFractionDigits: 0 });
    pushMsg({ role: 'agent', text: viaClick
      ? `Region set around that point (≈ ${km2} km²) — right-drag to sweep a bigger box.`
      : `Region set (≈ ${km2} km²).` });
  }, [pushMsg]);
  const onMapClick = useCallback((_lng: number, _lat: number) => { /* left-click inspects features */ }, []);

  // ---- LIVE: drive the real agent over SSE, updating the map from events ----
  const runLive = useCallback(async (text: string) => {
    let agentIdx = -1;
    setMessages((prev) => { const base = [...prev, { role: 'user', text } as ChatMessage]; agentIdx = base.length; return [...base, { role: 'agent', streaming: true, trace: [] }]; });
    const patch = (up: Partial<ChatMessage>) => setMessages((prev) => { if (agentIdx < 0 || agentIdx >= prev.length) return prev; const n = prev.slice(); n[agentIdx] = { ...n[agentIdx], ...up }; return n; });
    mapLayerDelivered.current = false;
    const trace: TraceLine[] = [];
    const addTrace = (t: TraceLine) => { trace.push(t); patch({ trace: [...trace] }); };
    setBusy(true);
    // A drawn region becomes spatial context for the agent (the API has no geometry
    // field, so it rides along in the prompt). The user bubble keeps the original text.
    const hints: string[] = [];
    if (spatial && drawnRegion) hints.push(`Focus on this drawn area — bbox [${polygonBBox(drawnRegion).map((n) => n.toFixed(4)).join(', ')}] (minLon,minLat,maxLon,maxLat, EPSG:4326).`);
    if (spatial && uploadContext.current) hints.push(uploadContext.current);
    const regionHint = hints.length ? `\n\n(${hints.join(' ')})` : '';
    try {
      abortRef.current = new AbortController();
      const res = await streamChat(text + regionHint, {
        signal: abortRef.current.signal,
        threadId: threadRef.current, memoryId: memoryRef.current,
        fileIds: pendingFileIds.current, agentDev: true,
        includeMcpTools: spatial,
        enabledSearchMethods: spatial ? null : CHAT_ONLY_METHODS,
      }, asAgentConfig(), {
        onTrace: (l) => addTrace(l),
        onToolCall: (name, args) => {
          addTrace({ text: `${name}(${short(args)})`, kind: 'tool' });
          drawFromToolArgs(name, args);
        },
        onToolResult: (name, parsed) => {
          const fc = extractFeatures(parsed);
          if (fc.features.length) putLayer({ kind: 'geojson', id: `live-${name}`, source: 'kb', label: `${name} (preview)`, data: fc, style: { fill: [124, 58, 237, 180], line: [255, 255, 255, 255], pointRadius: 6, lineWidth: 2 } });
        },
        onFile: (files) => patch({ artifacts: files }),
        onMapLayer: async (layer) => {
          mapLayerDelivered.current = true;
          if (layer.url) layerSourceFiles.current.add(fileKey(layer.url));
          // A raster (embedding PCA image, segmentation mask) is an IMAGE draped over its
          // footprint. It must be handled before the GeoJSON path below, which would try to
          // parse the PNG as JSON, fail, and silently deliver nothing.
          if (layer.render === 'raster' && layer.url && layer.bounds) {
            const bounds = layer.bounds;
            putLayer({
              kind: 'raster', id: layer.id, source: (layer.source as any) || 'analysis',
              label: layer.label, url: resolveUrl(layer.url), bounds,
              opacity: layer.opacity ?? 0.85, fitBounds: true,
              // Kept, not dropped: this branch names every field it copies, so a descriptor
              // field left out here never reaches the layer list or the session store.
              ...(layer.embedding ? { embedding: layer.embedding } : {}),
            });
            fitView(bboxToFC(bounds));
            addTrace({ text: `map: raster — ${layer.label}`, kind: 'tool' });
            return;
          }
          // A layer may arrive inline or as a URL to fetch (large heatmaps/choropleths).
          let fc = layer.geojson;
          if (!fc && layer.url) {
            try {
              const res = await fetch(resolveUrl(layer.url));
              if (res.ok) fc = await res.json();
            } catch { /* leave it undelivered rather than guess */ }
          }
          if (!fc || !Array.isArray(fc.features) || !fc.features.length) return;
          const heat = layer.render === 'heatmap';
          const green = layer.source === 'overpass';
          // A categorical layer (LISA classes, Gi* bands, region ids) must keep that render:
          // collapsing it to 'geojson' sends a class-NAME style column down the numeric
          // choropleth ramp, which yields NaN for every feature and one flat fill.
          const categorical = layer.render === 'categories' && !!layer.legend?.length;
          putLayer({
            kind: 'geojson', id: layer.id, source: (layer.source as any) || 'analysis', label: layer.label,
            data: fc, fitBounds: true, sourceUrl: layer.url,
            render: heat ? 'heatmap' : categorical ? 'categories' : 'geojson',
            styleBy: layer.styleBy,
            legend: categorical ? layer.legend : undefined,
            // Boundary-only, for a layer whose job is to frame what is drawn beneath it.
            // Dropped here once already: the flag reached the client and died in this call,
            // so the zone came back as a violet slab over its own pixel image.
            outline: layer.outline === true ? true : undefined,
            partial: layer.sampled && layer.total
              ? { shown: layer.count ?? fc.features.length, total: layer.total }
              : undefined,
            style: heat
              ? { opacity: 0.85 }
              : green
                ? { fill: [16, 185, 129, 120], line: [16, 185, 129, 255], lineWidth: 3, pointRadius: 5 }
                : { fill: [124, 58, 237, 120], line: [124, 58, 237, 255], lineWidth: 2, pointRadius: 6 },
          });
          fitView(fc);
          addTrace({ text: `map: ${layer.render || 'layer'} — ${layer.label} `
            + (layer.sampled && layer.total
                ? `(SAMPLE: ${layer.count ?? fc.features.length} of ${layer.total})`
                : (() => { const n = layer.count ?? fc.features.length;
                            return `(${n} feature${n === 1 ? '' : 's'})`; })()), kind: 'tool' });
        },
        onIds: ({ threadId, memoryId }) => { if (threadId) threadRef.current = threadId; if (memoryId) memoryRef.current = memoryId; },
      });
      pendingFileIds.current = []; // attached to the thread server-side now
      // Reconcile with authoritative terminal result.
      const authoritative = extractFeatures(res.response);
      if (authoritative.features.length) {
        putLayer({ kind: 'geojson', id: 'agent-results', source: 'kb', label: 'Agent results', data: authoritative, style: { fill: [124, 58, 237, 220], line: [255, 255, 255, 255], pointRadius: 7, lineWidth: 2 }, fitBounds: true });
        fitView(authoritative);
      }
      const html = res.error ? '' : renderMarkdown(res.answer || '_(no answer text)_', resolveUrl);
      patch({ html, text: res.error ? `⚠ ${res.error}` : undefined, artifacts: res.downloads, response: res.response, trace: [...trace], streaming: false });
      // Only when the agent did NOT place a layer itself. AWAITED, not fire-and-forget: the
      // snapshot in `finally` used to run first, so a turn whose layers came from the artifact
      // fallback was saved with an empty layer list and restored as a bare transcript.
      if (!mapLayerDelivered.current) await loadVectorArtifacts(res.downloads);
    } catch (e: any) {
      const stopped = e?.name === 'AbortError';
      // An auth refusal is not a failed request and must not read like one: "Request failed:
      // Forbidden" tells someone nothing about what to do, and with the role gate starting at
      // contributor this is the message most accounts will actually see.
      const text = stopped
        ? '⏹ Stopped. Anything already on the map stays; ask me something else.'
        : e instanceof AuthError ? authMessage(e) : `Request failed: ${e.message}`;
      patch({ text, streaming: false });
    } finally { setBusy(false); abortRef.current = null; snapshotSession(); }
  }, [asAgentConfig, putLayer, fitView, resolveUrl, spatial, drawnRegion, loadVectorArtifacts]);

  const drawFromToolArgs = useCallback((name: string, args: any) => {
    if (!args || typeof args !== 'object') return;
    const bbox = args.bbox || args.bounding_box || args.extent;
    if (Array.isArray(bbox) && bbox.length === 4 && bbox.every((n: any) => typeof n === 'number')) {
      const region = bboxPolygon([bbox[0], bbox[1]], [bbox[2], bbox[3]]);
      setDrawnRegion(region); fitView({ type: 'FeatureCollection', features: [{ type: 'Feature', geometry: region, properties: {} }] });
    } else if (args.geometry && args.geometry.type) {
      putLayer({ kind: 'geojson', id: `live-region-${name}`, source: 'analysis', label: `${name} region`, data: { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: args.geometry, properties: {} }] }, style: { fill: [255, 127, 14, 40], line: [255, 127, 14, 255], lineWidth: 2 } });
    }
  }, [fitView, putLayer]);

  // ---- LOCAL: the deterministic mock (offline fallback) ----
  const runLocal = useCallback(async (text: string) => {
    pushMsg({ role: 'user', text });
    const intent = parseIntent(text);
    if (intent.kind === 'clear') { setLayers([]); pushMsg({ role: 'agent', text: 'Cleared all layers.' }); return; }
    if (intent.kind === 'help') { pushMsg({ role: 'agent', text: 'Offline demo. Try: “find flood datasets”, “show hospitals here”, “features that intersect the upload”, “buffer 2 km”, “heatmap”. Switch to the live agent (⚙ Connection) for the full toolkit.' }); return; }
    if (intent.kind === 'kb') {
      const region = intent.useRegion ? (drawnRegion ?? viewportPolygon()) : drawnRegion;
      const filter = region ? { geometry: region as Geometry, relation: 'intersects' as const } : null;
      const hits: KbHit[] = searchKb(intent.query, filter);
      const { centroids, boxes } = kbHitsToFeatureCollections(hits);
      putLayer({ kind: 'geojson', id: 'kb-boxes', source: 'kb', label: 'KB footprints', data: boxes, style: { fill: [124, 58, 237, 40], line: [124, 58, 237, 220], lineWidth: 1.5 } });
      putLayer({ kind: 'geojson', id: 'kb-centroids', source: 'kb', label: 'KB results', data: centroids, style: { fill: [124, 58, 237, 235], line: [255, 255, 255, 255], pointRadius: 7 }, fitBounds: true });
      if (hits.length) fitView(centroids);
      pushMsg({ role: 'agent', text: hits.length ? `Found ${hits.length}:\n${hits.slice(0, 5).map((h) => `• ${h.title}`).join('\n')}` : `No matches for “${intent.query}”.` });
      return;
    }
    if (intent.kind === 'overpass') {
      const region = drawnRegion ?? viewportPolygon();
      if (!region) { pushMsg({ role: 'agent', text: 'Draw a region or zoom in and say “here”.' }); return; }
      setBusy(true);
      try { const fc = await queryOverpass(intent.preset, polygonBBox(region));
        putLayer({ kind: 'geojson', id: `overpass-${intent.preset.key}`, source: 'overpass', label: `OSM: ${intent.preset.label}`, data: fc, style: { fill: [16, 185, 129, 230], line: [16, 185, 129, 255], lineWidth: 2, pointRadius: 5 } });
        pushMsg({ role: 'agent', text: `Pulled ${fc.features.length} ${intent.preset.label}. ${stats(fc, areaKm2(region))}` });
      } catch (e: any) { pushMsg({ role: 'agent', text: `Overpass failed: ${e.message}` }); } finally { setBusy(false); }
      return;
    }
    if (intent.kind === 'relate') {
      const target = pickTarget(layersRef.current, intent.targetHint);
      if (!target) { pushMsg({ role: 'agent', text: 'Upload a GeoJSON or pull a layer first.' }); return; }
      setBusy(true);
      try { const fc = await queryOverpass(intent.preset, layerBBox(target.data));
        const sel = selectRelated(fc, target.data, intent.relation); const id = `relate-${intent.preset.key}-${target.id}`;
        putLayer({ kind: 'geojson', id, source: 'analysis', label: `${intent.preset.label} ∩ ${target.label}`, data: sel.fc, style: { fill: [37, 99, 235, 90], line: [37, 99, 235, 255], lineWidth: 3, pointRadius: 5 }, fitBounds: true });
        fitView(sel.fc.features.length ? sel.fc : target.data);
        pushMsg({ role: 'agent', text: `${fc.features.length} ${intent.preset.label} found; ${sel.fc.features.length} intersect “${target.label}”.` });
      } catch (e: any) { pushMsg({ role: 'agent', text: `Failed: ${e.message}` }); } finally { setBusy(false); }
      return;
    }
    if (intent.kind === 'analysis') {
      const target = pickTarget(layersRef.current, intent.targetHint);
      if (!target) { pushMsg({ role: 'agent', text: 'No layer to analyze yet.' }); return; }
      if (intent.op === 'heatmap') { putLayer({ ...target, id: `${target.id}-heat`, label: `heatmap(${target.label})`, render: 'heatmap' }); pushMsg({ role: 'agent', text: `Heatmap of ${target.label}.` }); return; }
      let out; let id;
      if (intent.op === 'buffer') { out = bufferFC(target.data, intent.km); id = `analysis-buffer-${target.id}`; }
      else if (intent.op === 'hull') { out = convexHull(target.data); id = `analysis-hull-${target.id}`; }
      else { const region = drawnRegion ?? viewportPolygon(); if (!region) { pushMsg({ role: 'agent', text: 'Draw a region to clip to.' }); return; } out = clipToRegion(target.data, region as Geometry); id = `analysis-clip-${target.id}`; }
      putLayer({ kind: 'geojson', id, source: 'analysis', label: `${intent.op}(${target.label})`, data: out.fc, style: { fill: [245, 158, 11, 70], line: [245, 158, 11, 255], lineWidth: 2, pointRadius: 5 } });
      pushMsg({ role: 'agent', text: out.summary });
      return;
    }
    pushMsg({ role: 'agent', text: `Not sure what to map for “${text}”. Try KB search, an OSM entity, a relation, or an analysis.` });
  }, [drawnRegion, putLayer, pushMsg, fitView]);

  const runAgent = useCallback((text: string) => (mode === 'live' ? runLive(text) : runLocal(text)), [mode, runLive, runLocal]);

  const onUpload = useCallback(async (files: File[]) => {
    // Always try to show GeoJSON on the map locally for instant feedback.
    for (const f of files) {
      if (/\.(geo)?json$/i.test(f.name)) {
        try { const j = JSON.parse(await f.text());
          const fc: FeatureCollection = j.type === 'FeatureCollection' ? j : j.type === 'Feature' ? { type: 'FeatureCollection', features: [j] } : { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: j, properties: {} }] };
          const name = f.name.replace(/\.(geo)?json$/i, '');
          putLayer({ kind: 'geojson', id: `upload-${name}`, source: 'upload', label: `Upload: ${name}`, data: fc, style: { fill: [239, 68, 68, 90], line: [239, 68, 68, 255], pointRadius: 6 }, fitBounds: true });
          fitView(fc);
          // Remember the upload's extent so the agent can bound Overpass by "the geojson I uploaded".
          try { const bb = layerBBox(fc); uploadContext.current = `The uploaded file "${name}" covers bbox [${bb.map((x) => x.toFixed(4)).join(', ')}] (minLon,minLat,maxLon,maxLat, EPSG:4326).`; } catch { /* */ }
        } catch { /* not parseable geojson */ }
      }
    }
    if (mode === 'live') {
      try {
        const recs: FileRecord[] = await uploadFiles(files, asAgentConfig());
        pendingFileIds.current.push(...recs.map((r) => r.file_id));
        sessionFileIds.current.push(...recs.map((r) => r.file_id));
        // The preview layer above was built from the local File, so it has no url and could
        // not survive a reload. Now that the same bytes live in the file store, record where
        // to re-fetch them so a restored session shows the upload too.
        for (const r of recs) {
          const stem = (r.filename || '').replace(/\.(geo)?json$/i, '');
          if (stem && r.download_url) {
            setLayers((prev) => prev.map((l) => (
              l.id === `upload-${stem}` && l.kind === 'geojson'
                ? { ...l, sourceUrl: r.download_url } : l)));
          }
        }
        pushMsg({ role: 'agent', text: `Uploaded ${recs.length} file(s) to the agent: ${recs.map((r) => r.filename).join(', ')}. They're attached to this conversation — ask me about them.` });
      } catch (e: any) { pushMsg({ role: 'agent', text: `Upload to agent failed: ${e.message}` }); }
    } else {
      pushMsg({ role: 'agent', text: `Loaded ${files.length} file(s) onto the map.` });
    }
  }, [mode, asAgentConfig, putLayer, fitView, pushMsg]);

  return (
    <div className={`app ${mapVisible ? 'map-on' : 'chat-only'}${resizing ? ' resizing' : ''}`}>
      <TopNav demoMode={demoMode || tokenMode} me={tokenMode ? me : null}
        onToggleSettings={() => setShowSettings((s) => !s)}
        onToggleHistory={() => { setShowHistory((v) => !v); void refreshSessions(); }}
        sessionCount={sessions.length}
        tab={tab}
        onSetTab={(t) => {
          setTab(t);
          // The remote-sensing tab is ABOUT the map: there is no drawing a region on a map
          // that is not on screen, so opening the tab opens the map. Leaving it does not
          // close the map again — by then the user may have put something on it.
          if (t !== 'rs') return;
          setMapVisible(true);
          // Land on the subject — but only when there is nothing of the user's to displace.
          // Yanking the view away from a layer they just produced, or a region they drew,
          // would be worse than starting them somewhere generic. fitView carries the waiting:
          // the map may not have loaded yet, and it re-fits once the pane finishes expanding.
          if (!layers.length && !drawnRegion) fitView(bboxToFC(CHAMPAIGN_URBANA));
        }} />
      {showHistory && (
        <div className="history" role="dialog" aria-label="Past conversations">
          <div className="history-head">
            <strong>Past conversations</strong>
            <button className="hbtn" onClick={startNewSession}>New conversation</button>
            <button className="hbtn" onClick={() => setShowHistory(false)}>Close</button>
          </div>
          {/* Where they are kept is DIFFERENT in token mode, and the account badge two inches
              away already promises that conversations follow the account to another browser.
              The old line said the opposite of that, to the same person, on the same screen. */}
          {!sessions.length && (
            <p className="hempty">
              No saved conversations yet.{' '}
              {tokenMode
                ? 'They will be saved to your I-GUIDE account and follow you to another browser.'
                : 'They are kept in this browser only.'}
            </p>
          )}
          <ul className="hlist">
            {sessions.map((s2) => (
              <li key={s2.id} className={s2.id === sessionIdRef.current ? 'hrow current' : 'hrow'}>
                <button className="hopen" onClick={() => void restoreSession(s2.id)} title="Open and continue">
                  <span className="htitle">{s2.title}</span>
                  <span className="hmeta">
                    {new Date(s2.updatedAt).toLocaleString()} · {s2.messageCount} messages
                    {s2.layerCount ? ` · ${s2.layerCount} layers` : ''}
                    {s2.fileCount ? ` · ${s2.fileCount} files` : ''}
                  </span>
                </button>
                <button className="hdel" title="Delete this conversation"
                  onClick={() => void deleteSession(s2.id).then(() => refreshSessions())}>×</button>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="workspace" style={{ '--chat-w': `${chatW}px` } as CSSProperties}>
        {/* MOUNT the map only while it is shown. Hiding it with display:none left it mounted
            at ~40x30 and MapLibre never fired `load`: window.__map stayed unset and the
            canvas painted nothing (observed 400x300 inside an 820x646 container). */}
        {mapVisible && (
          <div className="mapwrap" id="mapwrap" ref={mapBoxRef}>
            {/* The layer manager floats over the MAP, so it has to be positioned against the map
                pane. As a sibling of .mapwrap its containing block was .workspace — the full
                width of BOTH panes — so a narrow map pushed it across the seam and over the
                transcript, where its z-index 5 beats a static .chat. .mapwrap is already
                position:relative, so this costs no CSS. */}
            <LeftPanel
              layers={layers} selected={selected}
              onToggleLayer={toggleLayer} onRemoveLayer={removeLayerById}
              onFitLayer={fitLayer} onSetOpacity={setLayerOpacity} onClearSelection={() => setSelected(null)}
            />
            {mapBoxReady && (
              <AgentMap
                layers={layers} drawnRegion={drawnRegion} drawPreview={drawPreview}
                onMapClick={onMapClick} onHover={() => {}} onFeatureClick={onFeatureClick}
                onReady={() => {
                  const fc = pendingFit.current;
                  pendingFit.current = null;
                  if (fc) fitView(fc);
                }}
                onResize={refitAfterResize}
                onRegionPreview={onRegionPreview}
                onRegionDrawn={onRegionDrawn}
              />
            )}
          </div>
        )}
        {/* Only while there are two panes to divide. In chat-only the chat is a centred reading
            column, and a handle beside it is dead chrome that would also shift the `margin:0 auto`
            centring by half its width. */}
        {mapVisible && (
          <Splitter
            chatMin={CHAT_W_MIN} mapMin={MAP_W_MIN} defaultWidth={CHAT_W_DEFAULT}
            onResizeStart={() => { resizingRef.current = true; setResizing(true); }}
            onResizeEnd={(px) => {
              resizingRef.current = false;
              setResizing(false);
              setChatW(px);
              // One settle pass, once the browser has laid the new width out. The per-frame refit
              // was suppressed above; and a fitBounds still easing when the drag began was framed
              // for the pre-drag width — MapLibre skips its own stop() for the whole 800ms of an
              // ease, so nothing else would ever correct it.
              requestAnimationFrame(() => {
                const m = (window as any).__map;
                if (m) { try { m.resize(); } catch { /* */ } }
                refitAfterResize();
              });
            }}
          />
        )}
        <ChatPanel
          messages={messages} busy={busy} tab={tab} hasRegion={!!drawnRegion} layers={layers}
          mapVisible={mapVisible} onToggleMap={() => setMapVisible((v) => !v)}
          models={models}
          mode={mode} cfg={cfg} spatial={spatial} showSettings={showSettings && !demoMode && !tokenMode} resolveUrl={resolveUrl}
          onSend={runAgent}
        onStop={() => abortRef.current?.abort()}
          onClearRegion={() => { setDrawnRegion(null); pushMsg({ role: 'agent', text: 'Region cleared.' }); }}
          onUpload={onUpload}
          onSetMode={setMode} onSetCfg={setCfg}
          onSetSpatial={setSpatial}
        />
      </div>
    </div>
  );
}

function short(v: any): string { try { const s = typeof v === 'string' ? v : JSON.stringify(v); return s.length > 80 ? s.slice(0, 80) + '…' : s; } catch { return ''; } }

type VectorLayer = Extract<LayerArtifact, { kind: 'geojson' }>;
// Buffer / hull / clip / relate all read `.data`, so only a vector layer can be their
// target. Narrowing here keeps every caller honest instead of guarding at each use.
function pickTarget(all: LayerArtifact[], hint: string): VectorLayer | null {
  const layers = all.filter((l): l is VectorLayer => l.kind === 'geojson');
  if (!layers.length) return null;
  const h = hint.toLowerCase();
  const pool = layers.filter((l) => l.source !== 'analysis');
  const base = pool.length ? pool : layers;
  if (/\b(upload|uploaded|geojson|my data|my layer)\b/.test(h)) { const up = [...base].reverse().find((l) => l.source === 'upload'); if (up) return up; }
  const byWord = base.find((l) => h.includes(l.source) || h.split(/\s+/).some((w) => w.length > 3 && l.label.toLowerCase().includes(w)));
  return byWord ?? base[base.length - 1];
}
