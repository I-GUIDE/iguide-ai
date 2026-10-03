"""Every image that installs requirements.txt installs it through constraints.txt.

``constraints.txt`` is the deployed agent-api image's own ``pip freeze``. It describes the next
image only while every ``pip install`` in the images reads it. Without ``-c constraints.txt``,
pip resolves each bare name in ``requirements.txt`` to that day's newest version: nine days after
the 2026-09-22 build, an unpinned resolve already differed from the running image on 42 of 175
packages. Nothing fails when an install leaves the lock out. The image just stops matching it.

These tests read the Dockerfiles as text, so they do not replace building one. What they add is
that an edit which would let an image float again fails on a checkout:

* a ``pip install`` without the lock, or one that runs before the lock is copied in;
* a package a Dockerfile installs by name (py-spy, the spaCy model) that the lock does not pin.
  CI installs only what ``requirements.txt`` asks for, so its drift step never sees these;
* ``spacy download``, which installs whatever model version spaCy's compatibility table names
  on build day, outside the lock;
* an image that installs rasterio or fiona without naming ``libexpat1``. Their wheels bundle
  GDAL but link the system's ``libexpat.so.1``, which python:3.11-slim lacks. The GDAL and QGIS
  packages that bring it in today are exactly the layers that look safe to drop.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / "constraints.txt"


# ------------------------------------------------------------------ reading a Dockerfile

def _instructions(text: str) -> list[tuple[str, str]]:
    """``(INSTRUCTION, arguments)`` per logical line. Backslash continuations are joined; comment
    and blank lines are dropped, including inside a continuation, which Docker allows."""
    out: list[tuple[str, str]] = []
    logical = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            logical += line[:-1] + " "
            continue
        name, _, rest = (logical + line).partition(" ")
        logical = ""
        out.append((name.upper(), rest.strip()))
    return out


def _without_flags(rest: str) -> str:
    """An instruction's arguments without its leading ``--flag`` options (``--chown``,
    ``--mount``)."""
    return re.sub(r"^(?:--\S+\s+)*", "", rest)


def _copy_sources(rest: str) -> list[str]:
    """Sources of a COPY or ADD, in the shell form or the JSON form, relative to the context."""
    operands = _without_flags(rest)
    args = json.loads(operands) if operands.startswith("[") else shlex.split(operands)
    return [posixpath.normpath(arg.lstrip("/")) for arg in args[:-1]]


def _commands(rest: str) -> list[list[str]]:
    """The commands a RUN instruction executes, each as an argv, split on the shell's command
    separators. Quoted text stays whole, so a ``;`` inside ``python -c "..."`` splits nothing."""
    operands = _without_flags(rest)
    if operands.startswith("["):
        return [json.loads(operands)]
    lexer = shlex.shlex(operands, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    commands: list[list[str]] = [[]]
    for token in lexer:
        if token in {"&&", "||", ";", "|", "&"}:
            commands.append([])
        else:
            commands[-1].append(token)
    return [command for command in commands if command]


def _pip_install_args(argv: list[str]) -> list[str] | None:
    """What follows ``install`` when ``argv`` is a pip install, however pip is invoked
    (``pip``, ``pip3``, ``python -m pip``, a full path); otherwise ``None``."""
    for index, token in enumerate(argv[:-1]):
        if re.fullmatch(r"pip(3(\.\d+)?)?", PurePosixPath(token).name) and argv[index + 1] == "install":
            return argv[index + 2:]
    return None


def _constraint_files(args: list[str]) -> list[str]:
    files: list[str] = []
    for index, arg in enumerate(args):
        if arg in {"-c", "--constraint"} and index + 1 < len(args):
            files.append(args[index + 1])
        elif arg.startswith("--constraint="):
            files.append(arg.split("=", 1)[1])
        elif arg.startswith("-c") and not arg.startswith("--") and len(arg) > 2:
            files.append(arg[2:])
    return files


# pip install options whose value is a separate argument, so the value is not read as a package.
_TAKES_VALUE = {
    "-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-i", "--index-url",
    "--extra-index-url", "-f", "--find-links", "-t", "--target", "--prefix", "--root",
}


def _named_requirements(args: list[str]) -> list[str]:
    """The requirements a pip install names on its command line, as opposed to in a file."""
    named: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg.startswith("-"):
            skip = arg in _TAKES_VALUE
        else:
            named.append(arg)
    return named


def _canonical(name: str) -> str:
    """A project name as pip compares them (PEP 503): case and runs of ``-_.`` do not matter."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_names(text: str) -> set[str]:
    """The projects constraints.txt pins, from its ``name==version`` and ``name @ url`` lines."""
    names: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.add(_canonical(re.split(r"\s*(?:==|@)", line, maxsplit=1)[0]))
    return names


def _apt_packages(text: str) -> set[str]:
    """Every package an ``apt-get install`` or ``apt install`` in a RUN instruction names."""
    packages: set[str] = set()
    for name, rest in _instructions(text):
        if name != "RUN":
            continue
        for argv in _commands(rest):
            if PurePosixPath(argv[0]).name in {"apt-get", "apt"} and "install" in argv:
                packages.update(arg for arg in argv[argv.index("install") + 1:] if not arg.startswith("-"))
    return packages


# ------------------------------------------------------------------ the check

def _problems(dockerfile: str, locked: set[str]) -> list[str]:
    """Each way the Dockerfile with this text installs Python packages around the lock."""
    problems: list[str] = []
    lock_copied = False
    for name, rest in _instructions(dockerfile):
        if name == "FROM":
            lock_copied = False
        elif name in {"COPY", "ADD"} and not any(arg.startswith("--from") for arg in rest.split()):
            lock_copied = lock_copied or "constraints.txt" in _copy_sources(rest)
        elif name == "RUN":
            for argv in _commands(rest):
                shown = " ".join(argv)
                if "spacy" in argv and "download" in argv[argv.index("spacy"):]:
                    problems.append(f"`{shown}` installs a model outside the lock; install it by name with -c constraints.txt")
                args = _pip_install_args(argv)
                if args is None:
                    continue
                if not any(PurePosixPath(path).name == "constraints.txt" for path in _constraint_files(args)):
                    problems.append(f"`{shown}` does not pass -c constraints.txt")
                elif not lock_copied:
                    problems.append(f"`{shown}` runs before constraints.txt is copied in")
                for requirement in _named_requirements(args):
                    if "://" in requirement or requirement.startswith((".", "/")):
                        problems.append(f"`{shown}` installs {requirement} by URL or path; pin it in constraints.txt and install it by name")
                        continue
                    project = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
                    if project is None or _canonical(project.group(0)) not in locked:
                        problems.append(f"`{shown}` installs {requirement}, which constraints.txt does not pin")
    return problems


def _installs_root_requirements(path: Path) -> bool:
    """Whether the image copies in the repository's own requirements.txt, not a service's."""
    return any(
        name in {"COPY", "ADD"} and "requirements.txt" in _copy_sources(rest)
        for name, rest in _instructions(path.read_text(encoding="utf-8"))
    )


# Every Dockerfile at the top two levels of the repository that installs requirements.txt. A new
# image is covered as soon as it copies that file in.
IMAGES = sorted(
    path for path in {*REPO.glob("Dockerfile*"), *REPO.glob("*/Dockerfile*")}
    if path.is_file() and _installs_root_requirements(path)
)
_IDS = [path.relative_to(REPO).as_posix() for path in IMAGES]


# ------------------------------------------------------------------ the images

def test_every_image_that_installs_requirements_is_checked():
    """If discovery found nothing, every test below would pass by checking nothing."""
    assert {"rag_pipeline/Dockerfile", "MCP_server/Dockerfile", "metadata-extraction-server/Dockerfile"} <= set(_IDS)


@pytest.mark.parametrize("dockerfile", IMAGES, ids=_IDS)
def test_every_pip_install_goes_through_the_lock(dockerfile):
    problems = _problems(dockerfile.read_text(encoding="utf-8"), _locked_names(LOCK.read_text(encoding="utf-8")))

    assert not problems, f"{dockerfile.relative_to(REPO)} installs around constraints.txt:\n" + "\n".join(
        f"  {problem}" for problem in problems
    )


@pytest.mark.parametrize("dockerfile", IMAGES, ids=_IDS)
def test_an_image_with_rasterio_or_fiona_names_libexpat1(dockerfile):
    requested = {
        _canonical(match.group(0))
        for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line.strip()))
    }
    if not {"rasterio", "fiona"} & requested:
        pytest.skip("requirements.txt no longer asks for rasterio or fiona")

    assert "libexpat1" in _apt_packages(dockerfile.read_text(encoding="utf-8")), (
        f"{dockerfile.relative_to(REPO)} installs rasterio/fiona but no apt layer names libexpat1; "
        "it is there today only as a dependency of other apt packages"
    )


# ------------------------------------------------------------------ the check itself

_LOCKED = {"py-spy", "en-core-web-sm", "numpy"}
_HEAD = "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt constraints.txt ./\n"


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        pytest.param(_HEAD + "RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt\n", [], id="through-the-lock"),
        pytest.param(
            "FROM python:3.11-slim\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n",
            ["does not pass -c constraints.txt"],
            id="unpinned-as-before-this-change",
        ),
        pytest.param(
            "FROM python:3.11-slim\nCOPY requirements.txt .\nRUN pip install -r requirements.txt -c constraints.txt\n"
            "COPY constraints.txt .\n",
            ["runs before constraints.txt is copied in"],
            id="lock-copied-too-late",
        ),
        pytest.param(
            "FROM python:3.11-slim AS build\nCOPY constraints.txt .\nFROM python:3.11-slim\n"
            "RUN pip install -c constraints.txt numpy\n",
            ["runs before constraints.txt is copied in"],
            id="lock-copied-in-another-stage",
        ),
        pytest.param(_HEAD + "RUN python -m spacy download en_core_web_sm\n", ["installs a model outside the lock"], id="spacy-download"),
        pytest.param(_HEAD + "RUN pip install --no-cache-dir -c constraints.txt py-spy\n", [], id="named-and-locked"),
        pytest.param(_HEAD + "RUN pip install -c constraints.txt En_Core_Web_SM\n", [], id="names-compare-as-pip-compares-them"),
        pytest.param(
            _HEAD + "RUN pip install --no-cache-dir -c constraints.txt flamegraph\n",
            ["installs flamegraph, which constraints.txt does not pin"],
            id="named-but-not-locked",
        ),
        pytest.param(
            _HEAD + "RUN apt-get update && python3 -m pip install --constraint=/app/constraints.txt \\\n    py-spy\n",
            [],
            id="python-m-pip-long-flag-continuation",
        ),
        pytest.param(_HEAD + "RUN pip3 install py-spy\n", ["does not pass -c constraints.txt"], id="pip3"),
        pytest.param(_HEAD + 'RUN ["pip", "install", "py-spy"]\n', ["does not pass -c constraints.txt"], id="exec-form"),
        pytest.param(
            _HEAD + "RUN pip install -c constraints.txt https://example.org/x-1.0-py3-none-any.whl\n",
            ["by URL or path"],
            id="by-url",
        ),
        pytest.param(
            _HEAD + 'RUN python -c "import sys; print(sys.version)" && pip install -c constraints.txt numpy\n',
            [],
            id="separator-inside-quotes",
        ),
        pytest.param(
            _HEAD + "# RUN pip install -r requirements.txt\nRUN pip install -r requirements.txt -c constraints.txt\n",
            [],
            id="commented-out",
        ),
    ],
)
def test_the_check_catches_each_way_around_the_lock(dockerfile, expected):
    """A guard is only as good as the wrong answers it rejects. Each case is a Dockerfile the
    check has to judge correctly; the second is the agent-api image's own install line before
    this change."""
    problems = _problems(dockerfile, _LOCKED)

    assert len(problems) == len(expected), problems
    for problem, fragment in zip(problems, expected):
        assert fragment in problem


@pytest.mark.parametrize(
    ("dockerfile", "declared"),
    [
        ("RUN apt-get update && apt-get install -y gdal-bin libexpat1 && rm -rf /var/lib/apt/lists/*\n", True),
        ("RUN apt-get update && apt-get install -y --no-install-recommends \\\n    libexpat1 \\\n    && true\n", True),
        ("RUN apt-get update && apt-get install -y gdal-bin libgdal-dev\n", False),
        ("RUN echo libexpat1\n", False),
        ("# RUN apt-get install -y libexpat1\nRUN apt-get install -y gdal-bin\n", False),
    ],
    ids=["named", "continued-lines", "only-as-a-dependency", "mentioned-not-installed", "commented-out"],
)
def test_the_apt_reader(dockerfile, declared):
    assert ("libexpat1" in _apt_packages(dockerfile)) is declared
