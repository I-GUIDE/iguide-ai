"""The Swagger page's spec builds, and every route's documentation is valid YAML.

flasgger parses every view's docstring after `---` as YAML to build `/apispec_1.json`. One
unparseable docstring fails the whole spec with a 500, so /apidocs/ renders nothing. That
happened on 2026-11-11: `/agent/examples` (Stage 73) documented a response as
`description: Not signed in (\`reason: examples_need_sign_in\` ...)`, and the unquoted `: ` inside
it is not valid YAML. It was live in production with nothing reporting it.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402


def test_every_documented_route_parses_as_yaml():
    broken = []
    for rule in server.app.url_map.iter_rules():
        doc = inspect.getdoc(server.app.view_functions.get(rule.endpoint)) or ""
        if "---" not in doc:
            continue
        try:
            yaml.safe_load(doc.split("---", 1)[1])
        except yaml.YAMLError as exc:
            broken.append(f"{rule.rule}: {str(exc).splitlines()[0]}")
    assert not broken, "route docs that break /apispec_1.json:\n" + "\n".join(broken)


def test_the_swagger_spec_builds():
    response = server.app.test_client().get("/apispec_1.json")
    assert response.status_code == 200
    assert "/agent/chat/stream" in response.get_json()["paths"]
