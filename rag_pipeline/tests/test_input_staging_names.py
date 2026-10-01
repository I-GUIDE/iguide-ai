"""Two inputs must never claim one name in the sandbox work dir.

`_build_staging` stages each file under its file_id AND its filename, and deduped only by
SOURCE. Two different files sharing a filename therefore emitted the same dest: the second copy
overwrote the first, `available_as` went on advertising both, and the tool description steers
the model straight at the colliding name — so the peer analysed the wrong dataset under the
right name. When neither file had a file_id it was worse: one file in /work and the other
unreachable under any name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import agent_runtime.langchain_exec_tools as lex


def _resolver(monkeypatch, mapping):
    """Stub resolution so these test the NAMING, not the allowed-root policy."""
    def _resolve(ref):
        return mapping[str(ref)]
    monkeypatch.setattr(lex, "_resolve_input_file", _resolve)


def test_two_files_sharing_a_filename_do_not_collide(monkeypatch):
    _resolver(monkeypatch, {
        "a": (Path("/store/file_aaa__data.csv"), {"file_id": "file_aaa", "filename": "data.csv", "size_bytes": 10}),
        "b": (Path("/store/file_bbb__data.csv"), {"file_id": "file_bbb", "filename": "data.csv", "size_bytes": 10}),
    })
    staging, staged, errors, _ = lex._build_staging(["a", "b"])
    assert not errors
    dests = [s["dest"] for s in staging]
    assert len(dests) == len(set(dests)), f"a name is claimed twice: {dests}"


def test_available_as_reports_only_the_names_the_file_really_has(monkeypatch):
    """The model is told to open these, so an alias it does not own is a lie that sends it to
    another dataset."""
    _resolver(monkeypatch, {
        "a": (Path("/store/file_aaa__data.csv"), {"file_id": "file_aaa", "filename": "data.csv", "size_bytes": 10}),
        "b": (Path("/store/file_bbb__data.csv"), {"file_id": "file_bbb", "filename": "data.csv", "size_bytes": 10}),
    })
    staging, staged, _errors, _ = lex._build_staging(["a", "b"])
    dests = {s["dest"] for s in staging}
    for entry in staged:
        for name in entry["available_as"]:
            assert name in dests, f"{name} advertised but never staged"
    # The LAST claimant keeps the plain name — refs arrive oldest-first, so the newest input
    # is the one the question is about. The earlier file is honest about not having it.
    assert "data.csv" not in staged[0]["available_as"]
    assert "data.csv" in staged[1]["available_as"]


def test_a_lone_file_gets_both_of_its_names(monkeypatch):
    """Stability matters: the common single-file case must be unchanged."""
    _resolver(monkeypatch, {
        "a": (Path("/store/file_aaa__data.csv"), {"file_id": "file_aaa", "filename": "data.csv", "size_bytes": 10}),
    })
    _staging, staged, _errors, _ = lex._build_staging(["a"])
    assert staged[0]["available_as"] == ["file_aaa", "data.csv"]


def test_a_file_with_no_id_still_gets_a_reachable_name(monkeypatch):
    """Two local paths sharing a basename: neither has a file_id, so the loser used to end up
    with no name at all and was unreachable."""
    _resolver(monkeypatch, {
        "acs/data.csv": (Path("/local/acs/data.csv"), None),
        "tiger/data.csv": (Path("/local/tiger/data.csv"), None),
    })
    staging, staged, errors, _ = lex._build_staging(["acs/data.csv", "tiger/data.csv"])
    assert not errors
    dests = [s["dest"] for s in staging]
    assert len(dests) == 2 and len(set(dests)) == 2, dests
    assert all(e["available_as"] for e in staged), "every input must be openable under some name"
    # Last claimant owns the plain name; with no file_id the earlier one still needs a name.
    assert staged[0]["available_as"] == ["data_2.csv"]
    assert staged[1]["available_as"] == ["data.csv"]


def test_the_same_source_twice_is_still_staged_once(monkeypatch):
    """The pre-existing source dedupe must survive: an id and its filename are one file."""
    _resolver(monkeypatch, {
        "file_aaa": (Path("/store/file_aaa__data.csv"), {"file_id": "file_aaa", "filename": "data.csv", "size_bytes": 10}),
        "data.csv": (Path("/store/file_aaa__data.csv"), {"file_id": "file_aaa", "filename": "data.csv", "size_bytes": 10}),
    })
    _staging, staged, _errors, _ = lex._build_staging(["file_aaa", "data.csv"])
    assert len(staged) == 1


def test_free_dest_keeps_the_extension():
    claimed = {"data.csv": "a"}
    assert lex._free_dest("data.csv", claimed) == "data_2.csv"
    assert lex._free_dest("data.csv", {"data.csv": "a", "data_2.csv": "b"}) == "data_3.csv"
    assert lex._free_dest("noextension", {"noextension": "a"}) == "noextension_2"
    assert lex._free_dest("archive.tar.gz", {"archive.tar.gz": "a"}) == "archive.tar_2.gz"


# --- who owns a contested filename ---------------------------------------------------------
#
# `refs` arrives oldest-first: the session's earlier files, then this turn's uploads
# (get_session_files returns ids oldest first). Handing the plain name to the FIRST claimant
# therefore gave it to a file from an earlier turn, and the peer opened the name it was given
# in the question and read the wrong dataset — a wrong answer with nothing on the surface.

def test_a_re_upload_under_the_same_name_owns_that_name(monkeypatch):
    """Turn 1 uploads data.csv; turn 3 uploads a DIFFERENT data.csv and asks about it."""
    _resolver(monkeypatch, {
        "file_old": (Path("/store/file_old__data.csv"),
                     {"file_id": "file_old", "filename": "data.csv", "size_bytes": 10}),
        "file_new": (Path("/store/file_new__data.csv"),
                     {"file_id": "file_new", "filename": "data.csv", "size_bytes": 10}),
    })
    _staging, staged, errors, _ = lex._build_staging(["file_old", "file_new"])
    assert not errors
    owners = [e["file_id"] for e in staged if "data.csv" in e["available_as"]]
    assert owners == ["file_new"], f"data.csv went to {owners}, not the upload in question"
    # the older file is still reachable, by the id that is unique to it
    assert staged[0]["available_as"] == ["file_old"]


def test_the_older_file_is_never_left_unreachable(monkeypatch):
    """Losing the filename contest must not mean losing every name."""
    _resolver(monkeypatch, {
        "file_old": (Path("/store/file_old__data.csv"),
                     {"file_id": "file_old", "filename": "data.csv", "size_bytes": 10}),
        "file_new": (Path("/store/file_new__data.csv"),
                     {"file_id": "file_new", "filename": "data.csv", "size_bytes": 10}),
    })
    staging, staged, _errors, _ = lex._build_staging(["file_old", "file_new"])
    dests = {s["dest"] for s in staging}
    for entry in staged:
        assert entry["available_as"], f"{entry['ref']} has no name at all"
        for name in entry["available_as"]:
            assert name in dests


def test_a_derived_name_does_not_steal_one_a_later_input_owns(monkeypatch):
    """The fallback search used to consider only names already handed out, so it could land on
    a filename a subsequent input actually has — and that input then lost its own name."""
    _resolver(monkeypatch, {
        "a/data.csv":   (Path("/A/data.csv"),   None),
        "b/data.csv":   (Path("/B/data.csv"),   None),
        "c/data_2.csv": (Path("/C/data_2.csv"), None),
    })
    staging, staged, errors, _ = lex._build_staging(["a/data.csv", "b/data.csv", "c/data_2.csv"])
    assert not errors
    dests = [s["dest"] for s in staging]
    assert len(dests) == len(set(dests)), dests
    # the input whose real name is data_2.csv keeps it
    assert staged[2]["available_as"] == ["data_2.csv"]
    # and the one that had to be renamed went past it
    assert staged[0]["available_as"] == ["data_3.csv"]


def test_three_files_sharing_one_filename_all_stay_reachable(monkeypatch):
    _resolver(monkeypatch, {
        r: (Path(f"/store/{r}__data.csv"), {"file_id": r, "filename": "data.csv", "size_bytes": 10})
        for r in ("file_a", "file_b", "file_c")
    })
    staging, staged, _errors, _ = lex._build_staging(["file_a", "file_b", "file_c"])
    dests = [s["dest"] for s in staging]
    assert len(dests) == len(set(dests)), dests
    assert [e["available_as"] for e in staged] == [
        ["file_a"], ["file_b"], ["file_c", "data.csv"]]
