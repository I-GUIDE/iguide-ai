"""The agent-api image must carry the page ``GET /agent/dashboard`` serves.

``api/server.py`` serves ``examples/agent_chat_stream_demo.html``, read relative to the
repository root, which is ``/app`` in the image. ``rag_pipeline/Dockerfile`` builds ``/app`` from
an explicit list of COPY instructions, and no version of it had named ``examples/``: on
2026-10-02 the route answered 500 (``FileNotFoundError``) at
``https://agent.i-guide.io/agent/dashboard`` and in an image built locally from ``prototype``,
while a checkout served the page.

The audit below works out where the page lands in the image by the Dockerfile's own rules
(``WORKDIR``, each ``COPY`` of the last stage, then ``.dockerignore``) and checks that it is
where the route reads it. It reads text, so it does not replace building the image. What it adds
is that dropping the COPY fails on a checkout, before anyone builds anything.

The Dockerfile and ``.dockerignore`` readers are copied from
``rag_pipeline/tests/test_image_skill_roots.py`` (I-GUIDE/iguide-ai#31, open when this was
written), whose ignore matcher was checked against a real build. One change: ``_lands_at``
takes ``file=True`` for a path that is a single file, because Docker places a file named as a
COPY source differently from a directory. Once both are merged the readers belong in one module.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path, PurePosixPath

import pytest

import api.server as server

# The route reads the page relative to the directory above its own package: the repository root
# on a checkout, /app in the image.
REPO_ROOT = Path(server.__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "rag_pipeline" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
PACKAGE = PurePosixPath(Path(server.__file__).resolve().parent.relative_to(REPO_ROOT).as_posix())
PAGE = PurePosixPath(server._DASHBOARD_PAGE.relative_to(REPO_ROOT).as_posix())


# ------------------------------------------------------------------ reading the Dockerfile

def _instructions(text: str) -> list[tuple[str, list[str]]]:
    """``(INSTRUCTION, args)`` per logical line. Backslash continuations are joined; comment and
    blank lines are dropped, including inside a continuation, which Docker allows."""
    out: list[tuple[str, list[str]]] = []
    logical = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            logical += line[:-1] + " "
            continue
        parts = (logical + line).split()
        logical = ""
        out.append((parts[0].upper(), parts[1:]))
    return out


def _copy_operands(args: list[str]) -> tuple[list[str], str] | None:
    """Sources and destination of a COPY or ADD, in the shell form or the JSON form."""
    operands = [arg for arg in args if not arg.startswith("--")]
    if operands and operands[0].startswith("["):
        try:
            operands = json.loads(" ".join(operands))
        except ValueError:
            return None
    if len(operands) < 2:
        return None
    return operands[:-1], operands[-1]


def _lands_at(path: PurePosixPath, instructions, *, file: bool = False) -> PurePosixPath | None:
    """Where a path of the build context ends up in the final image, or ``None`` if nothing
    copies it there.

    Follows WORKDIR and each COPY/ADD of the last stage; a later copy of the same path wins.
    Copying a directory copies its contents to the destination, so a path inside a copied
    directory (``.agents/skills`` inside ``.agents/``) lands at the destination plus its path
    relative to that directory.

    ``file=True`` says the path is a single file. A COPY that names that file as its source puts
    it inside a destination that ends in ``/`` (or is ``.``), and writes it AS any other
    destination, the way Docker does.
    """
    workdir = PurePosixPath("/")
    landed: PurePosixPath | None = None
    for name, args in instructions:
        if name == "FROM":
            workdir, landed = PurePosixPath("/"), None
        elif name == "WORKDIR" and args:
            workdir = workdir / args[0]
        elif name in {"COPY", "ADD"} and not any(arg.startswith("--from") for arg in args):
            operands = _copy_operands(args)
            if operands is None:
                continue
            sources, destination = operands
            for source in sources:
                try:
                    inside = path.relative_to(PurePosixPath(source.lstrip("/") or "."))
                except ValueError:
                    continue
                if file and inside == PurePosixPath("."):
                    into = destination.endswith("/") or destination == "."
                    landed = workdir / destination / path.name if into else workdir / destination
                else:
                    landed = workdir / destination / inside
    return landed


# ------------------------------------------------------------------ reading .dockerignore

def _ignore_regex(pattern: str) -> re.Pattern[str]:
    """The regular expression Docker compiles from one ``.dockerignore`` pattern
    (moby/patternmatcher). ``*`` and ``?`` never cross a ``/``; ``**`` crosses any number of
    directories and, at the end, matches everything below; ``[...]`` is a character class;
    ``\\`` escapes the next character. A pattern is anchored at the context root, so ``*.md``
    matches the root's markdown files and nothing deeper."""
    out = "^"
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                i += 1
                if i + 1 < n and pattern[i + 1] == "/":
                    i += 1
                out += ".*" if i + 1 == n else "(.*/)?"
            else:
                out += "[^/]*"
        elif ch == "?":
            out += "[^/]"
        elif ch in ".+()|{}$":
            out += "\\" + ch
        elif ch == "\\":
            i += 1
            out += "\\" + pattern[i] if i < n else "\\\\"
        else:
            out += ch
        i += 1
    return re.compile(out + "$")


def _ignore_patterns(text: str) -> list[tuple[bool, re.Pattern[str]]]:
    """``(is_exception, regex)`` per pattern, read the way the build reads the file: a line that
    starts with ``#`` is a comment, surrounding whitespace is dropped, the path is cleaned, and
    one leading ``/`` is removed because patterns are relative to the context root anyway."""
    patterns: list[tuple[bool, re.Pattern[str]]] = []
    for number, raw in enumerate(text.splitlines()):
        if number == 0:
            raw = raw.lstrip("﻿")
        if raw.startswith("#"):
            continue
        line = raw.strip()
        exception = line.startswith("!")
        if exception:
            line = line[1:].strip()
        if not line:
            continue
        line = posixpath.normpath("/" + line.lstrip("/") if line.startswith("/") else line)
        if len(line) > 1 and line.startswith("/"):
            line = line[1:]
        patterns.append((exception, _ignore_regex(line)))
    return patterns


def _excluded(path: str, patterns) -> bool:
    """Whether the build leaves ``path`` out of its context: true when the path or one of its
    parent directories matches a pattern, with the last matching pattern deciding and a ``!``
    pattern re-including."""
    parts = path.split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    excluded = False
    for exception, regex in patterns:
        if any(regex.match(candidate) for candidate in candidates):
            excluded = not exception
    return excluded


# ------------------------------------------------------------------ the audit

def _page_audit(dockerfile: str, dockerignore: str) -> str | None:
    """Why an image built by ``dockerfile`` would not serve the page, or ``None`` if it would."""
    instructions = _instructions(dockerfile)
    package = _lands_at(PACKAGE, instructions)
    if package is None:
        return f"{PACKAGE}/ is not copied into the image at all"
    read_at = package.parent / PAGE
    landed = _lands_at(PAGE, instructions, file=True)
    if landed is None:
        return "no COPY in the Dockerfile carries it"
    if landed != read_at:
        return f"the Dockerfile puts it at {landed}; the route reads {read_at}"
    if _excluded(PAGE.as_posix(), _ignore_patterns(dockerignore)):
        return f".dockerignore keeps {PAGE} out of the build context"
    return None


def test_the_image_carries_the_dashboard_page():
    """The guard. Dropping the COPY, sending the page elsewhere, or an ignore pattern that keeps it
    out of the build context fails here, on a checkout, without building anything."""
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8") if DOCKERIGNORE.exists() else ""
    problem = _page_audit(DOCKERFILE.read_text(encoding="utf-8"), dockerignore)

    assert problem is None, f"{PAGE} does not reach the image where /agent/dashboard reads it: {problem}"


def test_the_route_sends_the_file_the_audit_follows():
    """Ties the audit to the route: on a checkout, the file the audit follows into the image is
    the file ``/agent/dashboard`` sends."""
    response = server.app.test_client().get("/agent/dashboard")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert response.get_data() == (REPO_ROOT / PAGE).read_bytes()


_HEAD = "FROM python:3.11-slim\nWORKDIR /app\nCOPY api/ ./api/\n"
_COPY = "COPY examples/agent_chat_stream_demo.html ./examples/\n"


@pytest.mark.parametrize(
    ("dockerfile", "dockerignore"),
    [
        pytest.param(_HEAD + _COPY, "", id="this-fix"),
        pytest.param(_HEAD + "COPY examples/ ./examples/\n", "", id="the-whole-directory"),
        pytest.param("FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n", "", id="whole-context"),
        pytest.param(
            _HEAD + "COPY examples/agent_chat_stream_demo.html ./examples/agent_chat_stream_demo.html\n", "",
            id="named-destination",
        ),
        pytest.param(_HEAD + 'COPY ["examples/agent_chat_stream_demo.html", "./examples/"]\n', "", id="json-form"),
        pytest.param(
            _HEAD + "COPY --chown=1000:1000 examples/agent_chat_stream_demo.html ./examples/\n", "", id="with-a-flag",
        ),
        pytest.param("FROM python:3.11-slim\nWORKDIR /srv\nCOPY api/ ./api/\n" + _COPY, "", id="another-workdir"),
        # Docker anchors a pattern at the context root, so `*.html` drops only the root's own pages.
        pytest.param(_HEAD + _COPY, "*.html\n", id="a-root-pattern-does-not-reach-examples"),
    ],
)
def test_the_audit_accepts_each_correct_copy(dockerfile, dockerignore):
    assert _page_audit(dockerfile, dockerignore) is None


@pytest.mark.parametrize(
    ("dockerfile", "dockerignore", "reason"),
    [
        pytest.param(_HEAD, "", "no COPY", id="as-before-this-fix"),
        pytest.param(
            _HEAD + "COPY examples/agent_chat_stream_demo.html ./\n", "",
            "puts it at /app/agent_chat_stream_demo.html;", id="misplaced",
        ),
        pytest.param(
            _HEAD + "COPY examples/agent_chat_stream_demo.html ./examples\n", "",
            "puts it at /app/examples;", id="no-trailing-slash-writes-a-file-named-examples",
        ),
        pytest.param(
            "FROM python:3.11-slim AS build\nWORKDIR /app\n" + _COPY + _HEAD, "", "no COPY",
            id="copied-in-an-earlier-stage-only",
        ),
        pytest.param(_HEAD + "COPY examples/ ./examples/\n", "**/*.html\n", ".dockerignore", id="ignore-every-page"),
        pytest.param(_HEAD + _COPY, "examples\n", ".dockerignore", id="ignore-the-examples-directory"),
    ],
)
def test_the_audit_rejects_each_way_the_page_can_be_lost(dockerfile, dockerignore, reason):
    """A guard is only as good as the wrong answers it rejects. Every case in both lists was built
    for real on 2026-10-02, and each build agreed with the audit. Five of these six build without
    an error and leave a route that answers 500. The sixth, an ignore pattern that drops a file a
    COPY names, fails the build."""
    problem = _page_audit(dockerfile, dockerignore)

    assert problem is not None and reason in problem, problem
