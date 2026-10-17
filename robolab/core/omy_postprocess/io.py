"""Small durable, collision-refusing I/O helpers for offline artifacts."""

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path):
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def write_json(path, value):
    """Create a new JSON artifact; never replace an existing one."""
    with open(path, "x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_write_json(path, value):
    """Publish a single report without exposing a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses even an empty existing
        # destination. The temporary name is on the same filesystem.
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(temporary)


def publish_directory(temporary, destination):
    """Linux atomic directory publication with RENAME_NOREPLACE.

    A check followed by os.rename() can overwrite an empty directory created by
    another process in between. Fail closed when the kernel/filesystem cannot
    provide no-replace publication; the recoverable temporary artifact remains.
    """
    temporary, destination = Path(temporary), Path(destination)
    if temporary.parent.resolve() != destination.parent.resolve():
        raise ValueError("Publication must occur within one destination directory")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "Atomic no-replace publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    fsync_directory(destination.parent)
