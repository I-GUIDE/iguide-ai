import type { LayerArtifact } from '../contracts';

export interface SelectedFeature {
  name: string;
  layerLabel: string;
  properties: Record<string, any>;
}

interface Props {
  layers: LayerArtifact[];
  selected: SelectedFeature | null;
  onToggleLayer: (id: string) => void;
  onRemoveLayer: (id: string) => void;
  onFitLayer: (id: string) => void;
  onSetOpacity?: (id: string, opacity: number) => void;
  onClearSelection: () => void;
}

const SOURCE_COLORS: Record<string, string> = {
  kb: '#7c3aed', overpass: '#1aa37a', upload: '#d1495b', analysis: '#c98a1a',
};

const HIDDEN_KEYS = new Set(['geometry', 'geojson', 'contents-embedding']);

function fmtValue(v: any): string {
  if (v == null) return '';
  if (typeof v === 'object') { try { return JSON.stringify(v); } catch { return String(v); } }
  return String(v);
}

export function LeftPanel(p: Props) {
  return (
    <aside className="leftpanel">
      <section className="lp-layers">
        <h3>Map layers <span className="cnt">{p.layers.length}</span></h3>
        {p.layers.length === 0 && <p className="lp-hint">No layers yet. Ask the agent for features, or upload data.</p>}
        <ul>
          {p.layers.map((l) => {
            const visible = l.visible !== false;
            // A raster is one image: it has no feature count and cannot be a sample.
            const n = l.kind === 'geojson' ? (l.data?.features?.length ?? 0) : null;
            const partial = l.kind === 'geojson' ? l.partial : undefined;
            return (
              <li key={l.id} className={visible ? '' : 'off'}>
                <button className="eye" title={visible ? 'Hide' : 'Show'} onClick={() => p.onToggleLayer(l.id)}>{visible ? '👁' : '⦰'}</button>
                <span className="dot" style={{ background: SOURCE_COLORS[l.source] ?? '#888' }} />
                <button className="lname" title="Zoom to layer" onClick={() => p.onFitLayer(l.id)}>{l.label}</button>
                <span className="cnt" title={partial ? `sample of ${partial.total}` : undefined}>
                  {partial ? `${partial.shown}/${partial.total}` : (n ?? "img")}
                </span>
                <button className="x" title="Remove" onClick={() => p.onRemoveLayer(l.id)}>×</button>
                {l.kind === 'raster' && p.onSetOpacity && (
                  <label className="lp-opacity" title="Layer opacity">
                    <input type="range" min={0} max={1} step={0.05}
                      value={l.opacity ?? 0.85}
                      onChange={(e) => p.onSetOpacity!(l.id, Number(e.target.value))} />
                  </label>
                )}
                {/* A cluster map is unreadable without its key: "High-High" is a colour on the
                    map and nowhere else. Only categorical layers carry one. */}
                {l.kind === 'geojson' && !!l.legend?.length && (
                  <ul className="lp-legend">
                    {l.legend.map((e) => (
                      <li key={e.label}>
                        <span className="swatch" style={{ background: `rgba(${e.color[0]},${e.color[1]},${e.color[2]},${e.color[3] / 255})` }} />
                        <span className="llabel" title={e.label}>{e.label}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      <section className="lp-feature">
        <h3>Selected feature{p.selected && <button className="x" title="Clear" onClick={p.onClearSelection}>×</button>}</h3>
        {!p.selected && <p className="lp-hint">Click a feature on the map to see its details.</p>}
        {p.selected && (
          <div className="fdetail">
            <div className="ftitle">{p.selected.name}</div>
            <div className="fsub">{p.selected.layerLabel}</div>
            <table className="ftable">
              <tbody>
                {Object.entries(p.selected.properties)
                  .filter(([k, v]) => !HIDDEN_KEYS.has(k) && v != null && v !== '')
                  .map(([k, v]) => (
                    <tr key={k}><th>{k}</th><td>{fmtValue(v)}</td></tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </aside>
  );
}
