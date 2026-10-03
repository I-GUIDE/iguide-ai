"""Start a child process without forking the agent, on the platform where forking kills it.

On macOS a long-lived agent process stops being able to ``fork()`` once PROJ has read its
database and the address space has moved on. Measured 2026-10-01 on the maintainer's Mac: after
one turn's first reprojection, every child the local server started died of SIGSEGV *before
exec* — ``qgis_metric_buffer`` reported returncode -11 with empty output at its first step and
``execute_code`` "failed during dependency installation", while neither program ever ran. From
the crash reports and ``scripts/repro_macos_fork_crash.py``:

1. pyproj's bundled PROJ registers a ``pthread_atfork`` child handler on its first database
   lookup. In the child it empties its handle cache, ``sqlite3_close``-ing every proj.db handle
   that no live PROJ object still holds.
2. That PROJ links Apple's ``/usr/lib/libsqlite3``, whose database descriptor is already invalid
   in the child: SQLite logs "cannot fstat db file" and then "close(...) - Bad file descriptor",
   and Apple's ``sqlite3_log`` sends each message to ``os_log``.
3. libsystem_trace's own fork handler has reset its logging state, so the child's first
   ``os_log`` re-reads the logging preferences through a pointer into a shared mapping the child
   did not inherit. Whether that pointer lands on mapped memory depends on where the child maps
   the fresh copy: a short script maps it back at the same address and survives; a process that
   has freed memory below it since (500,000 small objects released after the reprojection is
   enough, because CPython returns their emptied arenas) maps it lower, and the read faults.

``posix_spawn`` starts the child without running any fork handler in this process, so none of
that happens. CPython already uses it on macOS, but only when the program is given as a path,
``close_fds`` is false and there is no ``cwd``, ``preexec_fn``, ``pass_fds``, new session or
identity change. :func:`run` arranges exactly that. It resolves the program the way subprocess
would, and passes ``close_fds=False``, which costs nothing here: PEP 446 makes every descriptor
Python opens non-inheritable, and a process with ``api.server`` imported and a turn's tools run
had nothing inheritable but 0, 1 and 2. A working directory is reached through ``/bin/sh``,
which changes into it and then ``exec``-s the command, so the pid, the exit status and a kill on
timeout all belong to the command rather than to a shell standing in front of it.

Everywhere else :func:`run` is plain :func:`subprocess.run`.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence, Union

StrPath = Union[str, "os.PathLike[str]"]

# `cwd=` on its own sends CPython back to fork, so the shell changes directory instead and then
# replaces itself with the command.
_CD_THEN_EXEC = 'cd -- "$1" && shift && exec "$@"'

# Each of these either makes CPython fork even with everything else arranged (and would bring the
# crash back without a word), or changes what `args` means. No caller needs any of them.
_UNSUPPORTED = frozenset({
    "close_fds", "executable", "extra_groups", "group", "pass_fds", "preexec_fn",
    "process_group", "shell", "start_new_session", "umask", "user",
})


def avoids_fork() -> bool:
    """Whether :func:`run` must start children without forking this process (macOS)."""
    return sys.platform == "darwin"


def run(args: Union[StrPath, Sequence[StrPath]], *, cwd: Optional[StrPath] = None,
        env: Optional[Mapping[str, str]] = None, **kwargs: Any) -> subprocess.CompletedProcess:
    """:func:`subprocess.run`, except that on macOS this process is never forked to do it.

    Takes what the agent uses — ``capture_output``, ``text``, ``timeout``, ``input``, ``check``,
    ``stdin``/``stdout``/``stderr`` — and returns and raises what subprocess does, including
    ``FileNotFoundError`` for a program or ``cwd`` that does not exist and ``TimeoutExpired``
    once the command has been killed.
    """
    unsupported = sorted(_UNSUPPORTED.intersection(kwargs))
    if unsupported:
        raise TypeError(f"fork_safe.run() does not take {', '.join(unsupported)}: on macOS each "
                        "would either fork this process or change what the arguments mean")
    if not avoids_fork():
        return subprocess.run(args, cwd=cwd, env=env, **kwargs)
    if isinstance(args, (str, bytes, os.PathLike)):
        args = [args]
    argv = [os.fspath(a) for a in args]
    program = _program_path(argv[0], env)
    if cwd is not None:
        directory = os.fspath(cwd)
        if not os.path.isdir(directory):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), directory)
        argv = ["/bin/sh", "-c", _CD_THEN_EXEC, "sh", directory, program, *argv[1:]]
        program = "/bin/sh"
    return subprocess.run(argv, executable=program, env=env, close_fds=False, **kwargs)


def _program_path(name: str, env: Optional[Mapping[str, str]]) -> str:
    """``name`` as a path, searched on ``env``'s PATH as subprocess would, or FileNotFoundError.

    posix_spawn does not search PATH, and CPython only takes it for a program given as a path,
    so a bare name has to be resolved here. Raising the error subprocess raises keeps every
    caller's ``except FileNotFoundError`` meaning "that program is not installed".
    """
    if os.path.dirname(name):
        return name
    found = shutil.which(name, path=os.pathsep.join(os.get_exec_path(env)))
    if found is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), name)
    return found


__all__ = ["avoids_fork", "run"]
