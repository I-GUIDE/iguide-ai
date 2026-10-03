"""The agent-api image must carry every skill root the registry reads.

``agent_runtime/skills.py`` discovers skill bundles under ``REPO_ROOT/skills`` and
``REPO_ROOT/.agents/skills``. A root that does not exist is skipped without an error, by design,
so nothing complains when one goes missing. ``rag_pipeline/Dockerfile`` builds ``/app`` from an
explicit list of COPY instructions, and neither root was ever on it: inside the deployed
container on 2026-10-01, ``SkillRegistry.discover()`` found 0 skills and reported 0 errors,
while the same commit on a checkout found 3.

The audit below builds the part of ``/app`` that holds the skill roots in a temporary
directory, by the Dockerfile's own rules, and runs the real discovery over it. It reads the
Dockerfile and ``.dockerignore`` as text, so it does not replace building the image. What it
adds is that an edit which would bring the defect back fails on a checkout, before anyone
builds anything.
"""

from __future__ import annotations

import json
import posixpath
import re
import shutil
from pathlib import Path, PurePosixPath

import pytest

from agent_runtime import skills as skills_module
from agent_runtime.skills import DEFAULT_SKILL_ROOTS, REPO_ROOT, SkillRegistry

DOCKERFILE = REPO_ROOT / "rag_pipeline" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
# skills.py sets REPO_ROOT to the parent of this package, wherever the package is installed.
PACKAGE = PurePosixPath(Path(skills_module.__file__).resolve().parent.relative_to(REPO_ROOT).as_posix())
ROOTS = [PurePosixPath(root.relative_to(REPO_ROOT).as_posix()) for root in DEFAULT_SKILL_ROOTS]


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


def _lands_at(path: PurePosixPath, instructions) -> PurePosixPath | None:
    """Where a path of the build context ends up in the final image, or ``None`` if nothing
    copies it there.

    Follows WORKDIR and each COPY/ADD of the last stage; a later copy of the same path wins.
    Copying a directory copies its contents to the destination, so a path inside a copied
    directory (``.agents/skills`` inside ``.agents/``) lands at the destination plus its path
    relative to that directory.
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

def _write_skill(root: Path, name: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Planted by the image packaging test.\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _image_audit(dockerfile: str, dockerignore: str, workdir: Path):
    """Build an image's skill roots under ``workdir`` and run discovery over them the way the
    container does. Returns ``(lost, missing)``: each default root the image loses, mapped to
    the reason, and the names of skills the checkout discovers that the image does not."""
    instructions = _instructions(dockerfile)
    ignore = _ignore_patterns(dockerignore)

    # A build context: this checkout's skill roots, plus one planted skill in each, so every
    # root has something to lose whether or not the repository keeps a skill in it today.
    context = workdir / "context"
    planted: dict[PurePosixPath, str] = {}
    for index, root in enumerate(ROOTS):
        if (REPO_ROOT / root).is_dir():
            shutil.copytree(REPO_ROOT / root, context / root)
        planted[root] = f"planted-skill-{index}"
        _write_skill(context / root, planted[root])

    # The build: each root copied to where the Dockerfile puts it, minus what .dockerignore
    # keeps out of the context.
    image = workdir / "image"
    for root in ROOTS:
        destination = _lands_at(root, instructions)
        if destination is None:
            continue
        for source in sorted((context / root).rglob("*")):
            if source.is_dir() or _excluded(source.relative_to(context).as_posix(), ignore):
                continue
            target = image / destination.relative_to("/") / source.relative_to(context / root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    # The container: REPO_ROOT is the parent of wherever the package landed, and discovery
    # runs with no explicit roots, as every production caller runs it.
    package = _lands_at(PACKAGE, instructions)
    assert package is not None, f"{PACKAGE}/ is not copied into the image at all"
    read_at = {root: package.parent / root for root in ROOTS}
    image_roots = {root: image / path.relative_to("/") for root, path in read_at.items()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(skills_module, "DEFAULT_SKILL_ROOTS", tuple(image_roots.values()))
        for name in ("AGENT_SKILL_PATHS", "AGENT_SKILLS_PATHS", "AGENT_SKILLS_ENABLED"):
            mp.delenv(name, raising=False)
        in_image = {skill.name: skill for skill in SkillRegistry.discover().list()}

    lost: dict[PurePosixPath, str] = {}
    for root in ROOTS:
        found = in_image.get(planted[root])
        if found is not None and found.source_root == image_roots[root].resolve():
            continue
        landed = _lands_at(root, instructions)
        skill_file = f"{root}/{planted[root]}/SKILL.md"
        if landed is None:
            lost[root] = "no COPY in the Dockerfile carries it"
        elif landed != read_at[root]:
            lost[root] = f"the Dockerfile puts it at {landed}; the registry reads {read_at[root]}"
        elif _excluded(skill_file, ignore):
            lost[root] = f".dockerignore keeps {skill_file} out of the build context"
        else:
            lost[root] = "it is copied, yet discovery found no skill in it"

    on_checkout = SkillRegistry.discover([context / root for root in ROOTS])
    missing = sorted({skill.name for skill in on_checkout.list()} - set(in_image))
    return lost, missing


def test_the_image_discovers_a_skill_from_every_root(tmp_path):
    """The guard. Dropping either COPY, moving a root, or an ignore pattern that swallows
    SKILL.md files fails here, on a checkout, without building anything."""
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8") if DOCKERIGNORE.exists() else ""
    lost, missing = _image_audit(DOCKERFILE.read_text(encoding="utf-8"), dockerignore, tmp_path)

    assert not lost, "a skill root does not reach the agent-api image:\n" + "\n".join(
        f"  {root}/: {why}" for root, why in lost.items()
    )
    assert not missing, f"skills the checkout discovers and the image does not: {missing}"


_HEAD = "FROM python:3.11-slim\nWORKDIR /app\nCOPY agent_runtime/ ./agent_runtime/\n"
_BOTH = "COPY skills/ ./skills/\nCOPY .agents/ ./.agents/\n"


@pytest.mark.parametrize(
    ("dockerfile", "dockerignore", "expected_lost"),
    [
        pytest.param(_HEAD + _BOTH, "", [], id="both-roots"),
        pytest.param(_HEAD, "", ["skills", ".agents/skills"], id="neither-root-as-before-this-fix"),
        pytest.param(_HEAD + "COPY .agents/ ./.agents/\n", "", ["skills"], id="agents-only-the-first-fix"),
        pytest.param(_HEAD + "COPY skills/ ./skills/\n", "", [".agents/skills"], id="skills-only"),
        pytest.param(_HEAD + "COPY skills/ ./skill/\nCOPY .agents/ ./.agents/\n", "", ["skills"], id="misplaced"),
        pytest.param("FROM python:3.11-slim\nWORKDIR /app\nCOPY . .\n", "", [], id="whole-context"),
        pytest.param(
            _HEAD + 'COPY ["skills/", "./skills/"]\nCOPY --chown=appuser .agents/ ./.agents/\n', "", [],
            id="json-form-and-flags",
        ),
        pytest.param(
            "FROM python:3.11-slim\nWORKDIR /srv\nCOPY agent_runtime/ ./agent_runtime/\n" + _BOTH, "", [],
            id="another-workdir",
        ),
        pytest.param(
            "FROM python:3.11-slim AS build\nWORKDIR /app\n" + _BOTH + _HEAD, "", ["skills", ".agents/skills"],
            id="copied-in-an-earlier-stage-only",
        ),
        pytest.param(_HEAD + _BOTH, "**/*.md\n", ["skills", ".agents/skills"], id="ignore-every-markdown"),
        pytest.param(_HEAD + _BOTH, ".agents\n", [".agents/skills"], id="ignore-the-agents-directory"),
    ],
)
def test_the_audit_catches_each_way_a_root_can_be_lost(tmp_path, dockerfile, dockerignore, expected_lost):
    """A guard is only as good as the wrong answers it rejects. The first fix for this defect
    (``f8f99ef`` on ``backend_swap``) copied ``.agents/`` alone, and the test written with it
    asked whether *any* root was copied, so it passed with one root of two. Every case here
    is a Dockerfile or ignore file the audit has to judge correctly."""
    lost, _missing = _image_audit(dockerfile, dockerignore, tmp_path)

    assert [str(root) for root in lost] == expected_lost


@pytest.mark.parametrize(
    ("ignore_text", "path", "expected"),
    [
        # `*` stops at `/`, so `*.md` is anchored at the root: the rule this repository relies on.
        ("*.md", "AGENTS.md", True),
        ("*.md", "skills/example-skill/SKILL.md", False),
        # `**/` reaches every directory: the one-line edit that would drop every skill.
        ("**/*.md", "skills/example-skill/SKILL.md", True),
        # An exception re-includes, and only what it names.
        ("*.md\n!README.md", "README.md", False),
        ("*.md\n!README.md", "AGENTS.md", True),
        # A directory pattern excludes everything under it, and only from the root.
        ("tests/", "tests/test_x.py", True),
        ("tests/", "rag_pipeline/tests/test_x.py", False),
        ("/tools/", "tools/probe.py", True),
        # A dotted directory is an ordinary name to the matcher.
        (".agents", ".agents/skills/some-skill/SKILL.md", True),
        ("skills", "skills/example-skill/SKILL.md", True),
        # Docker's own examples.
        ("*/temp*", "somedir/temporary.txt", True),
        ("*/temp*", "somedir/sub/temporary.txt", False),
        ("temp?", "tempa", True),
        ("temp?", "tempab", False),
        # Character classes, as in Go's filepath.Match.
        ("temp[ab]", "tempb", True),
        ("temp[ab]", "tempc", False),
        # A comment is not a pattern, even when it reads like one.
        ("# *.md\n\n   ", "AGENTS.md", False),
    ],
)
def test_the_ignore_matcher_follows_dockers_rules(ignore_text, path, expected):
    """The matcher is only worth having if it agrees with the build. Checked on 2026-10-01
    against a real build of this repository's context: of 390 paths on disk, it predicted
    exactly the 365 that Docker sent."""
    assert _excluded(path, _ignore_patterns(ignore_text)) is expected
