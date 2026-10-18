"""Admins can see and adjust one user's daily usage limits (Stage 74.1).

Modelled on the platform backend's admin routes (``PUT /api/users/:id/role`` in
``iguide-ue-backend/routes/users_v2.js``): the user is named in the path by their CILogon id,
URL-encoded, the change is a JSON body, and only a privileged role may call it. Here that is
role 2 (Admin) or 1 (Super admin), the backend's ``authorizeRole(Role.ADMIN)``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402
from agent_runtime import file_store, identity  # noqa: E402
from agent_runtime import usage_limit as ul  # noqa: E402

SHANMUKH = "http://cilogon.org/serverE/users/443511"
PATH = f"/agent/admin/users/{quote(SHANMUKH, safe='')}/usage"


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(file_store, "storage_root", lambda: tmp_path)
    monkeypatch.delenv("AGENT_DAILY_CREDIT_LIMIT", raising=False)


def _as(monkeypatch, role):
    user = identity.User(id="http://cilogon.org/serverA/users/10128", role=role) if role else None
    monkeypatch.setattr(server, "_require_user", lambda: user)
    return server.app.test_client()


@pytest.mark.parametrize("role", [3, 4, 8, None])
def test_only_admins_may_look_or_change(monkeypatch, role):
    client = _as(monkeypatch, role)
    assert client.get(PATH).status_code == 403
    r = client.put(PATH, json={"creditLimit": 10})
    assert r.status_code == 403 and r.get_json()["reason"] == "admin_required"
    assert ul.override_for(SHANMUKH) == {}


def test_the_encoded_cilogon_id_reaches_the_handler_intact(monkeypatch):
    ul.record(SHANMUKH, {"cost_coins": 1.25})
    body = _as(monkeypatch, 2).get(PATH).get_json()
    assert body["userId"] == SHANMUKH          # the `//` survived routing
    assert body["today"]["coins"] == pytest.approx(1.25)
    assert body["today"]["credits"] == pytest.approx(1.25)
    assert body["limits"] == {"credits": 3.0} and body["defaultLimits"] == {"credits": 3.0}


def test_an_admin_raises_one_users_credit_limit_and_it_applies(monkeypatch):
    for _ in range(2):
        ul.record(SHANMUKH, {"cost_coins": 1.0})
    ul.record(SHANMUKH, {"cost_usd": 1.5})
    assert ul.refusal(SHANMUKH) is not None
    r = _as(monkeypatch, 2).put(PATH, json={"creditLimit": 10})
    assert r.status_code == 200
    body = r.get_json()
    assert body["limits"] == {"credits": 10.0}
    assert body["override"]["credits"] == 10.0
    assert ul.refusal(SHANMUKH) is None
    assert ul.limits_for("http://cilogon.org/serverE/users/1") == {"credits": 3.0}


def test_unlimited_zero_and_back_to_default(monkeypatch):
    client = _as(monkeypatch, 1)
    assert client.put(PATH, json={"creditLimit": "unlimited"}).get_json()["limits"]["credits"] is None
    assert client.put(PATH, json={"creditLimit": 0}).get_json()["limits"]["credits"] == 0.0
    assert ul.refusal(SHANMUKH) is not None    # 0 blocks the user
    body = client.put(PATH, json={"creditLimit": None}).get_json()
    assert body["limits"] == {"credits": 3.0} and body["override"] == {}


@pytest.mark.parametrize("key", ["coinLimit", "usdLimit"])
def test_the_retired_per_currency_limits_are_refused_by_name(monkeypatch, key):
    r = _as(monkeypatch, 2).put(PATH, json={key: 10})
    assert r.status_code == 400 and "creditLimit" in r.get_json()["error"]
    assert ul.override_for(SHANMUKH) == {}


def test_an_override_written_before_credits_still_applies(tmp_path):
    import json
    ul.set_override(SHANMUKH, credits=1)            # creates the file and the user's key
    path = tmp_path / "usage_ledger" / "overrides.json"
    data = json.loads(path.read_text())
    (key,) = data
    data[key] = {"coins": -1.0, "usd": 5.0, "set_at": "2026-10-17T22:00:00-05:00"}
    path.write_text(json.dumps(data))
    assert ul.limits_for(SHANMUKH) == {"credits": None}     # its coin override was "unlimited"


def test_reset_today_clears_the_days_spending(monkeypatch):
    ul.record(SHANMUKH, {"cost_usd": 3.5})
    assert ul.refusal(SHANMUKH) is not None
    body = _as(monkeypatch, 2).put(PATH, json={"resetToday": True}).get_json()
    assert body["today"]["usd"] == 0 and ul.refusal(SHANMUKH) is None


@pytest.mark.parametrize("bad", [-1, "lots", True, float("nan")])
def test_a_bad_limit_is_refused_and_changes_nothing(monkeypatch, bad):
    r = _as(monkeypatch, 2).put(PATH, json={"creditLimit": bad})
    assert r.status_code == 400
    assert ul.override_for(SHANMUKH) == {}


def test_an_empty_change_is_refused(monkeypatch):
    assert _as(monkeypatch, 2).put(PATH, json={}).status_code == 400


def test_the_override_file_holds_no_user_id(tmp_path, monkeypatch):
    _as(monkeypatch, 2).put(PATH, json={"creditLimit": 5})
    text = (tmp_path / "usage_ledger" / "overrides.json").read_text()
    assert "cilogon" not in text and "443511" not in text


def test_an_unencoded_cilogon_id_in_the_path_also_survives(monkeypatch):
    ul.record(SHANMUKH, {"cost_coins": 0.5})
    r = _as(monkeypatch, 2).get(f"/agent/admin/users/{SHANMUKH}/usage")
    assert r.status_code == 200 and r.get_json()["userId"] == SHANMUKH
