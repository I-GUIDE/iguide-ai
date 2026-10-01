"""Tests for the map_layer descriptor boundary (agent_runtime/map_layers.py).

Every tool's layer crosses build_map_layer on its way to the client, so invariants
the client depends on are enforced here rather than trusted to each emitter.
"""
# --- a categorical layer without its palette must not reach the client as 'categories' ---
# Observed twice, from two different tools: render='categories' with no legend made the client
# fall back to the numeric ramp, Number("High-High") is NaN, and all 801 features drew as one
# flat fill while the answer text described five colours.

def _descriptor(**over):
    import json
    ml = {"url": "/agent/files/x/download", "label": "LISA clusters",
          "render": "categories", "style_by": "lisa_class", "count": 801}
    ml.update(over)
    return json.dumps({"map_layer": ml})


def test_categorical_layer_without_a_legend_is_downgraded():
    from agent_runtime.map_layers import build_map_layer

    out = build_map_layer("add_map_layer", _descriptor())
    assert out["render"] == "shapes", "must not claim to encode classes it cannot colour"
    assert out.get("legend_missing") is True


def test_categorical_layer_keeps_its_render_when_the_legend_is_usable():
    from agent_runtime.map_layers import build_map_layer

    out = build_map_layer("add_map_layer", _descriptor(
        legend=[{"label": "High-High", "color": [215, 25, 28, 200]},
                {"label": "Low-Low", "color": [44, 123, 182, 200]}]))
    assert out["render"] == "categories"
    assert [e["label"] for e in out["legend"]] == ["High-High", "Low-Low"]


def test_a_malformed_legend_counts_as_no_legend():
    from agent_runtime.map_layers import build_map_layer

    out = build_map_layer("add_map_layer", _descriptor(
        legend=[{"label": "High-High", "color": "red"}, {"color": [1, 2, 3, 4]}]))
    assert out["render"] == "shapes"
    assert "legend" not in out


def test_raster_layer_skips_the_geojson_quality_checks():
    """A raster's url is a PNG. The vector QA has nothing to say about an image, and a
    stricter QA would downgrade it to 'shapes' — which stops the client drawing it at all."""
    import json

    from agent_runtime.map_layers import build_map_layer

    out = build_map_layer("embed_region", json.dumps({"map_layer": {
        "url": "/agent/files/file_abc/download", "label": "gse embedding (PCA-RGB)",
        "render": "raster", "bounds": [-88.3, 40.1, -88.2, 40.2], "opacity": 0.6}}))
    assert out["render"] == "raster", "must not be downgraded"
    assert out["bounds"] == [-88.3, 40.1, -88.2, 40.2]
    assert out["opacity"] == 0.6
    assert "degenerate" not in out and "legend_missing" not in out


def test_raster_without_bounds_is_dropped_not_shipped():
    """Bounds are what place the image; without them there is nothing to draw."""
    import json

    from agent_runtime.map_layers import build_map_layer

    assert build_map_layer("embed_region", json.dumps(
        {"map_layer": {"url": "/agent/files/f/download", "render": "raster"}})) is None


def test_outline_survives_the_descriptor():
    """build_map_layer builds a FIXED dict, so every field a tool sets has to be named here
    or it is silently dropped. Cost of learning that: an agent flag and a client renderer that
    both worked, and a zone that still came back as a violet slab over its own pixel image."""
    import json

    from agent_runtime.map_layers import build_map_layers

    out = build_map_layers("embed_zones", json.dumps({
        "ok": True,
        "map_layer": {"url": "http://x/z.geojson", "label": "gse embedded zone 17031330100",
                      "render": "shapes", "outline": True, "source": "analysis", "count": 1},
    }))
    assert len(out) == 1
    assert out[0]["outline"] is True

    plain = build_map_layers("add_map_layer", json.dumps({
        "ok": True, "map_layer": {"url": "http://x/y.geojson", "render": "shapes"},
    }))
    assert plain[0]["outline"] is False, "a normal layer still fills"


def _write_geojson(path, geoms):
    import json as _json
    path.write_text(_json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"i": i}, "geometry": g}
                     for i, g in enumerate(geoms)]}))


def test_layers_for_artifacts_turns_written_geodata_into_a_map(tmp_path):
    """A CLI code peer has no add_map_layer — it writes files. Its geodata used to reach the
    user as a download link and nothing else, which is the one thing this platform says an
    analysis result must not be."""
    from agent_runtime.map_layers import layers_for_artifacts

    _write_geojson(tmp_path / "hospitals.geojson",
                   [{"type": "Point", "coordinates": [-87.6 + i / 100, 41.8]} for i in range(5)])
    _write_geojson(tmp_path / "tracts.geojson",
                   [{"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}])
    (tmp_path / "notes.txt").write_text("not geodata")
    _write_geojson(tmp_path / "empty.geojson", [])

    artifacts = [{"filename": n, "download_url": f"/files/id_{n}/download"}
                 for n in ("hospitals.geojson", "tracts.geojson", "notes.txt",
                           "empty.geojson", "never_written.geojson")]
    layers = layers_for_artifacts(tmp_path, artifacts)
    by_label = {layer["label"]: layer for layer in layers}

    assert set(by_label) == {"hospitals", "tracts"}, \
        "text is not a layer, an empty layer draws nothing, and a missing file is not one"
    assert by_label["hospitals"]["render"] == "points" and by_label["hospitals"]["count"] == 5
    assert by_label["tracts"]["render"] == "shapes"
    assert by_label["hospitals"]["url"] == "/files/id_hospitals.geojson/download"


def test_a_big_point_set_becomes_a_heatmap(tmp_path):
    """Same rule add_map_layer applies for render="auto", so a layer looks the same however
    it was produced — 3000 raw points drawn as circles is a solid blob."""
    from agent_runtime.map_layers import layers_for_artifacts

    _write_geojson(tmp_path / "pts.geojson",
                   [{"type": "Point", "coordinates": [-87.6, 41.8]} for _ in range(2500)])
    layers = layers_for_artifacts(
        tmp_path, [{"filename": "pts.geojson", "download_url": "/files/x/download"}])
    assert layers[0]["render"] == "heatmap"


def test_artifact_layers_cross_the_same_validation_as_a_tool_layer(tmp_path):
    """They are emitted through build_map_layers like everything else, so the descriptor
    invariants apply to them too rather than being trusted because a peer produced them."""
    import json as _json

    from agent_runtime.map_layers import build_map_layers, layers_for_artifacts

    _write_geojson(tmp_path / "out.geojson",
                   [{"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}])
    descriptors = layers_for_artifacts(
        tmp_path, [{"filename": "out.geojson", "download_url": "http://x/files/f_1/download"}])
    built = build_map_layers("claude_run", _json.dumps({"ok": True, "map_layers": descriptors}))

    assert len(built) == 1
    assert built[0]["kind"] == "map_layer" and built[0]["render"] == "shapes"
    assert built[0]["url"] == "http://x/files/f_1/download"
