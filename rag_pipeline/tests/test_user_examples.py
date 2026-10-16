"""Examples users author for the Spatial Analysis tab (architecture stage 73).

The rules under test: only a signed-in user creates; only the owner edits, deletes or shares; a
share link reaches signed-in users until it is revoked; only an admin publishes; every refusal of
an example the caller may not see is a 404, never a 403; and the limits hold without leaving
partial files behind. Offline: the store is a temporary directory and identity is a locally
signed JWT.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import jwt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import api.server as server  # noqa: E402

SECRET = "s" * 64
COOKIE = "jwt-access-token-dev"

# Platform ids are URLs (AGENTS.md "Ownership"); single-token fixtures hide id-handling bugs.
ALICE = "http://cilogon.org/serverE/users/137206"
BOB = "http://cilogon.org/serverE/users/999999"
CAROL = "http://cilogon.org/serverE/users/424242"
ADMIN = "http://cilogon.org/serverE/users/1"

STEPS = [{"label": "Join", "prompt": "Join the tracts to the counts."},
         {"label": "Map it", "prompt": "Show the rate as a choropleth."}]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_FILE_STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("AGENT_MODE", "token")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_SECRET", SECRET)
    monkeypatch.setenv("JWT_ACCESS_TOKEN_NAME", COOKIE)
    for var in ("DEMO_MODE", "AGENT_CHAT_API_KEY", "AGENT_TOKEN_STRICT", "AGENT_TOKEN_VERIFY",
                "AGENT_MIN_ROLE", "AGENT_EXAMPLES_MAX_PER_USER", "AGENT_EXAMPLES_MAX_FILES",
                "AGENT_EXAMPLES_MAX_MB", "AGENT_EXAMPLES_MAX_STEPS", "AGENT_EXAMPLES_PUBLISH_ROLE"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path / "store"


def client_as(user_id=None, role=4):
    client = server.app.test_client()
    if user_id:
        token = jwt.encode({"id": user_id, "role": role, "exp": int(time.time()) + 3600},
                           SECRET, algorithm="HS256")
        client.set_cookie(COOKIE, token, domain="localhost")
    return client


def definition(**over):
    return {"title": "Chicago crime rates", "description": "Rates by tract.",
            "category": "Hypothesis testing", "tags": ["rates", "Moran's I"],
            "steps": STEPS, **over}


def create(client, files=(("tracts.geojson", b'{"type":"FeatureCollection","features":[]}'),
                          ("counts.csv", b"tract,n\n1,3\n")), **over):
    data = {"example": json.dumps(definition(**over)),
            "files": [(io.BytesIO(body), name) for name, body in files]}
    return client.post("/agent/examples", data=data, content_type="multipart/form-data")


def made(client, **kw):
    res = create(client, **kw)
    assert res.status_code == 201, res.get_json()
    return res.get_json()


# --- creating ------------------------------------------------------------------------------

def test_a_signed_in_user_creates_an_example_with_files_and_steps(env):
    ex = made(client_as(ALICE))
    assert ex["id"].startswith("ex_") and ex["source"] == "user"
    assert ex["title"] == "Chicago crime rates" and ex["blurb"] == "Rates by tract."
    assert ex["steps"] == STEPS and ex["tags"] == ["rates", "Moran's I"]
    assert [(f["name"], f["bytes"]) for f in ex["files"]] == [("tracts.geojson", 42), ("counts.csv", 12)]
    assert ex["files"][1]["url"] == f"/agent/examples/{ex['id']}/files/counts.csv"
    assert ex["mine"] is True and ex["visibility"] == "private" and ex["shareToken"] is None
    # The owner's platform id names a person and never leaves the server.
    assert ALICE not in json.dumps(ex)
    assert (env / "examples" / "files" / ex["id"] / "counts.csv").read_bytes() == b"tract,n\n1,3\n"


def test_a_json_body_creates_an_example_with_no_files():
    res = client_as(ALICE).post("/agent/examples", json=definition())
    assert res.status_code == 201 and res.get_json()["files"] == []


def test_the_owner_lists_it_and_nobody_else_does():
    ex = made(client_as(ALICE))
    mine = client_as(ALICE).get("/agent/examples").get_json()
    assert [e["id"] for e in mine["mine"]] == [ex["id"]]
    assert mine["canCreate"] is True and mine["canPublish"] is False
    theirs = client_as(BOB).get("/agent/examples").get_json()
    assert theirs["mine"] == theirs["shared"] == theirs["published"] == []


@pytest.mark.parametrize("over, field", [
    ({"title": "  "}, "title"),
    ({"category": ""}, "category"),
    ({"steps": []}, "at least one step"),
    ({"steps": [{"label": "x", "prompt": ""}]}, "step 1's question"),
    ({"title": "t" * 121}, "title is longer"),
    ({"tags": "not, a, list, but, " + ", ".join(f"t{i}" for i in range(20))}, "tags"),
])
def test_an_invalid_example_is_refused_naming_the_field(over, field, env):
    res = create(client_as(ALICE), **over)
    assert res.status_code == 400 and field in res.get_json()["error"]
    assert not list((env / "examples" / "defs").glob("*.json"))


def test_a_step_with_no_label_is_labelled_by_its_question():
    ex = made(client_as(ALICE), steps=[{"label": "", "prompt": "Count the hospitals within 5 km."}])
    assert ex["steps"][0]["label"] == "Count the hospitals within 5 km."


def test_outside_token_mode_nobody_can_create_but_published_examples_list(monkeypatch):
    admin_ex = made(client_as(ADMIN, role=2))
    assert client_as(ADMIN, role=2).post(f"/agent/examples/{admin_ex['id']}/publish").status_code == 200
    monkeypatch.setenv("AGENT_MODE", "dev")
    anon = client_as()
    listing = anon.get("/agent/examples").get_json()
    assert [e["id"] for e in listing["published"]] == [admin_ex["id"]]
    assert listing["canCreate"] is False
    res = create(anon)
    assert res.status_code == 403 and res.get_json()["reason"] == "examples_need_sign_in"


def test_a_signed_out_caller_in_token_mode_is_told_to_sign_in():
    res = client_as().get("/agent/examples")
    assert res.status_code == 403 and res.get_json()["reason"] == "not_signed_in"


# --- the owner, and everyone else ------------------------------------------------------------

def test_another_user_gets_404_for_every_route_on_a_private_example():
    ex = made(client_as(ALICE))
    bob = client_as(BOB)
    eid = ex["id"]
    for res in (bob.get(f"/agent/examples/{eid}"),
                bob.put(f"/agent/examples/{eid}", json=definition(title="mine now")),
                bob.delete(f"/agent/examples/{eid}"),
                bob.get(f"/agent/examples/{eid}/files/counts.csv"),
                bob.post(f"/agent/examples/{eid}/share"),
                bob.post(f"/agent/examples/{eid}/copy"),
                bob.delete(f"/agent/examples/{eid}/publish")):
        assert res.status_code == 404, res.request.path
    # ...and the answer is the one an id that never existed gets.
    assert bob.get("/agent/examples/ex_0000000000000000").get_json() == bob.get(f"/agent/examples/{eid}").get_json()
    assert client_as(ALICE).get(f"/agent/examples/{eid}").get_json()["title"] == "Chicago crime rates"


def test_the_owner_edits_steps_and_files_in_one_request(env):
    alice = client_as(ALICE)
    ex = made(alice)
    data = {"example": json.dumps(definition(title="Renamed", steps=list(reversed(STEPS)),
                                             keepFiles=["tracts.geojson"])),
            "files": [(io.BytesIO(b"new"), "extra.txt")]}
    res = alice.put(f"/agent/examples/{ex['id']}", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, res.get_json()
    body = res.get_json()
    assert body["title"] == "Renamed" and body["steps"] == list(reversed(STEPS))
    assert [f["name"] for f in body["files"]] == ["tracts.geojson", "extra.txt"]
    folder = env / "examples" / "files" / ex["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["extra.txt", "tracts.geojson"]


def test_an_edit_without_keepfiles_keeps_every_file():
    alice = client_as(ALICE)
    ex = made(alice)
    body = alice.put(f"/agent/examples/{ex['id']}", json=definition(title="x")).get_json()
    assert [f["name"] for f in body["files"]] == ["tracts.geojson", "counts.csv"]


def test_the_owner_deletes_it_with_its_files(env):
    alice = client_as(ALICE)
    ex = made(alice)
    assert alice.delete(f"/agent/examples/{ex['id']}").get_json() == {"deleted": True, "id": ex["id"]}
    assert alice.get(f"/agent/examples/{ex['id']}").status_code == 404
    assert not (env / "examples" / "files" / ex["id"]).exists()


def test_a_file_name_cannot_walk_out_of_the_example(env):
    alice = client_as(ALICE)
    ex = made(alice, files=(("../../escape.txt", b"x"),))
    assert [f["name"] for f in ex["files"]] == ["escape.txt"]
    assert not (env / "escape.txt").exists()
    assert alice.get(f"/agent/examples/{ex['id']}/files/..%2F..%2Fmetadata").status_code == 404
    assert alice.get(f"/agent/examples/{ex['id']}/files/escape.txt").data == b"x"


# --- sharing ---------------------------------------------------------------------------------

def test_a_share_link_reaches_another_signed_in_user_until_it_is_revoked():
    alice, bob = client_as(ALICE), client_as(BOB)
    ex = made(alice)
    eid = ex["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    assert token and alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"] == token

    opened = bob.post("/agent/examples/open", json={"token": token})
    assert opened.status_code == 200
    view = opened.get_json()
    assert view["mine"] is False and view["visibility"] == "link"
    assert "shareToken" not in view            # only the owner sees the token
    assert [e["id"] for e in bob.get("/agent/examples").get_json()["shared"]] == [eid]
    assert bob.get(f"/agent/examples/{eid}/files/counts.csv").data == b"tract,n\n1,3\n"
    # Seeing is not editing.
    assert bob.put(f"/agent/examples/{eid}", json=definition(title="hijacked")).status_code == 404
    assert bob.delete(f"/agent/examples/{eid}").status_code == 404

    # Carol never opened the link, so she cannot reach the example by its id.
    assert client_as(CAROL).get(f"/agent/examples/{eid}").status_code == 404

    assert alice.delete(f"/agent/examples/{eid}/share").status_code == 200
    assert bob.get("/agent/examples").get_json()["shared"] == []
    assert bob.get(f"/agent/examples/{eid}/files/counts.csv").status_code == 404
    assert bob.post("/agent/examples/open", json={"token": token}).status_code == 404


def test_rotating_the_link_cuts_off_the_old_one():
    alice, bob = client_as(ALICE), client_as(BOB)
    eid = made(alice)["id"]
    old = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    bob.post("/agent/examples/open", json={"token": old})
    new = alice.post(f"/agent/examples/{eid}/share", json={"rotate": True}).get_json()["shareToken"]
    assert new != old
    assert bob.get(f"/agent/examples/{eid}").status_code == 404
    assert client_as(CAROL).post("/agent/examples/open", json={"token": old}).status_code == 404
    assert client_as(CAROL).post("/agent/examples/open", json={"token": new}).status_code == 200


def test_a_share_link_needs_a_signed_in_user(monkeypatch):
    alice = client_as(ALICE)
    eid = made(alice)["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    assert client_as().post("/agent/examples/open", json={"token": token}).status_code == 403
    monkeypatch.setenv("AGENT_MODE", "dev")
    res = client_as().post("/agent/examples/open", json={"token": token})
    assert res.status_code == 403 and res.get_json()["reason"] == "examples_need_sign_in"


def test_the_owner_opening_their_own_link_does_not_list_it_as_shared():
    alice = client_as(ALICE)
    eid = made(alice)["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    assert alice.post("/agent/examples/open", json={"token": token}).get_json()["mine"] is True
    listing = alice.get("/agent/examples").get_json()
    assert [e["id"] for e in listing["mine"]] == [eid] and listing["shared"] == []


def test_a_shared_example_can_be_taken_off_the_list():
    alice, bob = client_as(ALICE), client_as(BOB)
    eid = made(alice)["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    bob.post("/agent/examples/open", json={"token": token})
    assert bob.delete(f"/agent/examples/{eid}/shared").get_json() == {"removed": True, "id": eid}
    assert bob.get("/agent/examples").get_json()["shared"] == []
    assert bob.delete(f"/agent/examples/{eid}/shared").status_code == 404
    assert alice.get(f"/agent/examples/{eid}").status_code == 200


def test_a_shared_example_is_copied_into_the_viewers_own_examples(env):
    alice, bob = client_as(ALICE), client_as(BOB)
    eid = made(alice)["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    bob.post("/agent/examples/open", json={"token": token})
    res = bob.post(f"/agent/examples/{eid}/copy")
    assert res.status_code == 201
    copy = res.get_json()
    assert copy["id"] != eid and copy["mine"] is True and copy["visibility"] == "private"
    assert copy["copiedFrom"] == eid and copy["steps"] == STEPS
    # The copy is Bob's own: it outlives the original and its link.
    alice.delete(f"/agent/examples/{eid}")
    assert bob.get(f"/agent/examples/{copy['id']}/files/counts.csv").data == b"tract,n\n1,3\n"
    assert bob.put(f"/agent/examples/{copy['id']}", json=definition(title="Bob's")).status_code == 200


# --- publishing ------------------------------------------------------------------------------

def test_only_an_admin_publishes_and_then_everyone_sees_it():
    alice = client_as(ALICE)
    eid = made(alice)["id"]
    assert alice.post(f"/agent/examples/{eid}/publish").status_code == 403   # owner, role 4
    admin = client_as(ADMIN, role=2)
    # An admin publishes what they can see; a private example of someone else is not that.
    assert admin.post(f"/agent/examples/{eid}/publish").status_code == 404
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    admin.post("/agent/examples/open", json={"token": token})
    res = admin.post(f"/agent/examples/{eid}/publish")
    assert res.status_code == 200 and res.get_json()["visibility"] == "published"
    assert admin.get("/agent/examples").get_json()["canPublish"] is True

    carol = client_as(CAROL)
    listing = carol.get("/agent/examples").get_json()
    assert [e["id"] for e in listing["published"]] == [eid]
    assert carol.get(f"/agent/examples/{eid}/files/tracts.geojson").status_code == 200
    # In one group only: the admin who opened the link sees it as published, not also shared.
    admin_listing = admin.get("/agent/examples").get_json()
    assert [e["id"] for e in admin_listing["published"]] == [eid] and admin_listing["shared"] == []
    # Published and still the owner's alone to change.
    assert carol.put(f"/agent/examples/{eid}", json=definition(title="x")).status_code == 404
    assert carol.delete(f"/agent/examples/{eid}/publish").status_code == 403


def test_the_publish_role_is_configurable(monkeypatch):
    monkeypatch.setenv("AGENT_EXAMPLES_PUBLISH_ROLE", "4")
    alice = client_as(ALICE)
    eid = made(alice)["id"]
    assert alice.post(f"/agent/examples/{eid}/publish").status_code == 200


def test_an_owners_edit_unpublishes_and_the_owner_can_withdraw():
    alice, admin = client_as(ALICE), client_as(ADMIN, role=2)
    eid = made(alice)["id"]
    token = alice.post(f"/agent/examples/{eid}/share").get_json()["shareToken"]
    admin.post("/agent/examples/open", json={"token": token})
    admin.post(f"/agent/examples/{eid}/publish")
    edited = alice.put(f"/agent/examples/{eid}", json=definition(title="Changed after approval")).get_json()
    assert edited["published"] is False and edited["unpublishedByEdit"]
    assert client_as(CAROL).get("/agent/examples").get_json()["published"] == []

    admin.post(f"/agent/examples/{eid}/publish")
    assert alice.delete(f"/agent/examples/{eid}/publish").get_json()["published"] is False


def test_an_admin_editing_their_own_published_example_keeps_it_published():
    admin = client_as(ADMIN, role=2)
    eid = made(admin)["id"]
    admin.post(f"/agent/examples/{eid}/publish")
    assert admin.put(f"/agent/examples/{eid}", json=definition(title="y")).get_json()["published"] is True


# --- limits ----------------------------------------------------------------------------------

def test_data_over_the_size_limit_is_refused_and_nothing_is_kept(monkeypatch, env):
    monkeypatch.setenv("AGENT_EXAMPLES_MAX_MB", "1")
    big = b"x" * (1024 * 1024 + 1)
    res = create(client_as(ALICE), files=(("small.csv", b"a"), ("big.tif", big)))
    assert res.status_code == 413 and "1 MB" in res.get_json()["error"]
    assert not list((env / "examples" / "defs").glob("*.json"))
    files_root = env / "examples" / "files"
    assert not files_root.exists() or not any(files_root.rglob("*"))


def test_an_edit_counts_the_kept_files_against_the_size_limit(monkeypatch, env):
    monkeypatch.setenv("AGENT_EXAMPLES_MAX_MB", "1")
    alice = client_as(ALICE)
    ex = made(alice, files=(("a.bin", b"x" * 700_000),))
    data = {"example": json.dumps(definition()), "files": [(io.BytesIO(b"y" * 400_000), "b.bin")]}
    res = alice.put(f"/agent/examples/{ex['id']}", data=data, content_type="multipart/form-data")
    assert res.status_code == 413
    folder = env / "examples" / "files" / ex["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["a.bin"]
    assert [f["name"] for f in alice.get(f"/agent/examples/{ex['id']}").get_json()["files"]] == ["a.bin"]


def test_too_many_files_or_steps_are_refused(monkeypatch):
    monkeypatch.setenv("AGENT_EXAMPLES_MAX_FILES", "2")
    monkeypatch.setenv("AGENT_EXAMPLES_MAX_STEPS", "2")
    alice = client_as(ALICE)
    res = create(alice, files=(("a", b"1"), ("b", b"2"), ("c", b"3")))
    assert res.status_code == 413 and "2 files" in res.get_json()["error"]
    res = create(alice, steps=STEPS + [{"label": "3", "prompt": "three"}])
    assert res.status_code == 400 and "2 steps" in res.get_json()["error"]


def test_one_account_keeps_a_bounded_number_of_examples(monkeypatch):
    monkeypatch.setenv("AGENT_EXAMPLES_MAX_PER_USER", "2")
    alice = client_as(ALICE)
    made(alice)
    eid = made(alice)["id"]
    assert create(alice).status_code == 413
    # Copies count too, and another account has its own allowance.
    assert alice.post(f"/agent/examples/{eid}/copy").status_code == 413
    assert create(client_as(BOB)).status_code == 201
