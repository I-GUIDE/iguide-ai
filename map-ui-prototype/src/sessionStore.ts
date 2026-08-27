/**
 * Local chat-session history: past conversations, their map layers, and enough identity to
 * carry one on.
 *
 * IndexedDB rather than localStorage. A transcript holds rendered answer HTML and a trace per
 * turn, so a handful of sessions passes localStorage's ~5 MB ceiling, and localStorage writes
 * synchronously on the main thread — which would stutter the map mid-stream.
 *
 * Layers are stored as DESCRIPTORS with their source url, never as geometry. A 15 MB heatmap
 * would otherwise be duplicated into the browser's storage on every turn; instead the file
 * store keeps it (retention is disabled server-side, so the url stays good) and restore
 * re-fetches through the same path the live `map_layer` event uses.
 *
 * Nothing here is authoritative. The server already owns conversational memory keyed by
 * `memoryId`; this is the client remembering which conversations exist, which it currently
 * forgets on reload. That is also why the shape is deliberately server-shaped: moving it behind
 * a per-user endpoint later means swapping the transport, not the record.
 */

import type { FeatureCollection } from 'geojson';
import type { LayerArtifact } from './contracts';
import type { ChatMessage } from './components/ChatPanel';

const DB_NAME = 'iguide-map-ui';
const DB_VERSION = 1;
const STORE = 'sessions';

/** A stored layer is restorable one of two ways: a `sourceUrl` to re-fetch (the usual case,
 *  and the only sane one for a 15 MB heatmap), or — for a layer delivered inline with no url
 *  behind it — its own small geometry. One of the two must be present or the layer is lost. */
export type StoredLayer = Omit<Extract<LayerArtifact, { kind: 'geojson' }>, 'data'> & {
  data?: FeatureCollection;
  sourceUrl?: string;
} | Extract<LayerArtifact, { kind: 'raster' }>;

export interface StoredSession {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  /** Agent-side identity: replaying these continues the conversation rather than starting one. */
  threadId: string;
  memoryId?: string | null;
  messages: ChatMessage[];
  layers: StoredLayer[];
  /** EVERY file uploaded in this session, not just the last turn's. The analysis tools only
   *  load when files are attached, so a session restored without these silently loses them. */
  fileIds: string[];
  region?: unknown;
  model?: string;
  provider?: string;
}

export type SessionSummary = Pick<StoredSession,
  'id' | 'title' | 'createdAt' | 'updatedAt' | 'threadId'> & {
  messageCount: number;
  layerCount: number;
  fileCount: number;
};

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const os = db.createObjectStore(STORE, { keyPath: 'id' });
        os.createIndex('updatedAt', 'updatedAt');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then((db) => new Promise<T>((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = run(t.objectStore(STORE));
    req.onsuccess = () => resolve(req.result as T);
    req.onerror = () => reject(req.error);
    t.oncomplete = () => db.close();
  }));
}

/** Drop geometry and any transient fields; keep what restore needs. */
/** Above this, inline geometry is dropped rather than stored: a layer this big came from a
 *  file and has a `sourceUrl` to re-fetch, and IndexedDB is not the place for 15 MB of
 *  coordinates. Inline-DELIVERED layers (overpass_search, spatial_search) arrive in the SSE
 *  event itself, so they are small by construction and have no url at all — dropping their
 *  geometry made them silently unrestorable, which read as "history lost my layers". */
const INLINE_KEEP_BYTES = 2_000_000;

export function toStoredLayer(a: LayerArtifact): StoredLayer {
  if (a.kind === 'raster') return { ...a };
  const { data, ...rest } = a as Extract<LayerArtifact, { kind: 'geojson' }> & {
    sourceUrl?: string;
  };
  if (rest.sourceUrl) return { ...rest, data: undefined };   // re-fetchable: store the pointer
  // No url: geometry is the only copy. Keep it if it is small enough to be worth keeping.
  let keep: typeof data | undefined;
  try {
    if (data && JSON.stringify(data).length <= INLINE_KEEP_BYTES) keep = data;
  } catch { /* unserialisable — treat as too big */ }
  return { ...rest, data: keep };
}

/** First user message, trimmed — the same thing a person would call the conversation. */
export function titleFor(messages: ChatMessage[]): string {
  const first = messages.find((m) => m.role === 'user' && (m.text || '').trim());
  const raw = (first?.text || '').trim().replace(/\s+/g, ' ');
  if (!raw) return 'New conversation';
  return raw.length > 72 ? `${raw.slice(0, 71)}…` : raw;
}

export async function saveSession(s: StoredSession): Promise<void> {
  try {
    await tx('readwrite', (store) => store.put(s) as unknown as IDBRequest<void>);
  } catch (err) {
    // History is a convenience: never let a storage failure break the conversation itself.
    console.warn('[sessions] save failed', err);
  }
}

export async function listSessions(): Promise<SessionSummary[]> {
  try {
    const all = await tx<StoredSession[]>('readonly', (store) => store.getAll() as IDBRequest<StoredSession[]>);
    return (all || [])
      .map((s) => ({
        id: s.id, title: s.title, createdAt: s.createdAt, updatedAt: s.updatedAt,
        threadId: s.threadId,
        messageCount: (s.messages || []).length,
        layerCount: (s.layers || []).length,
        fileCount: (s.fileIds || []).length,
      }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch (err) {
    console.warn('[sessions] list failed', err);
    return [];
  }
}

export async function loadSession(id: string): Promise<StoredSession | null> {
  try {
    return (await tx<StoredSession>('readonly', (store) => store.get(id) as IDBRequest<StoredSession>)) || null;
  } catch (err) {
    console.warn('[sessions] load failed', err);
    return null;
  }
}

export async function deleteSession(id: string): Promise<void> {
  try {
    await tx('readwrite', (store) => store.delete(id) as unknown as IDBRequest<void>);
  } catch (err) {
    console.warn('[sessions] delete failed', err);
  }
}

export function newSessionId(): string {
  const rand = Math.random().toString(36).slice(2, 10);
  return `sess-${Date.now().toString(36)}-${rand}`;
}
