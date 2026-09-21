"""Server-local episode controls, separate from the WebRTC keyboard."""

import os
import select
import struct
import sys
import termios
import tty


class LocalControls:
    """Read a Linux USB keyboard (evdev), or an explicitly selected terminal."""

    EVENT = struct.Struct("llHHI")
    KEYS = {49: "N", 19: "R", 35: "H", 48: "B", 1: "ESC"}

    def __init__(self, callbacks, device=None, source="evdev"):
        self.callbacks = callbacks
        self.source = source
        self.fd = None
        self.terminal_settings = None
        self.buffer = b""
        self.held = set()
        if source == "evdev":
            if not device:
                raise ValueError("--control-device must name the server's USB keyboard event device")
            try:
                self.fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot read local control device {device}: {exc}. Check device path/read permission."
                ) from exc
        elif source == "terminal":
            if not sys.stdin.isatty():
                raise ValueError("Terminal controls require a foreground TTY")
            self.fd = sys.stdin.fileno()
            self.terminal_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            raise ValueError(f"Unknown local control source: {source}")

    def _dispatch(self, key):
        callback = self.callbacks.get(key)
        if callback is not None:
            callback()

    def feed(self, data):
        """Decode chunks; process transitions once, ignoring key repeat."""
        self.buffer += data
        while len(self.buffer) >= self.EVENT.size:
            _, _, kind, code, value = self.EVENT.unpack(self.buffer[: self.EVENT.size])
            self.buffer = self.buffer[self.EVENT.size :]
            if kind == 0 and code == 3:
                raise RuntimeError("Local keyboard event overflow; restart collection")
            if kind != 1:
                continue
            if value == 0:
                self.held.discard(code)
            elif value == 1 and code not in self.held:
                self.held.add(code)
                self._dispatch(self.KEYS.get(code))

    def poll(self):
        if self.fd is None:
            return
        for _ in range(8):
            if not select.select([self.fd], [], [], 0)[0]:
                break
            try:
                data = os.read(self.fd, self.EVENT.size * 64 if self.source == "evdev" else 1)
            except BlockingIOError:
                break
            if not data:
                raise RuntimeError("Local control device disconnected")
            if self.source == "evdev":
                self.feed(data)
            else:
                self._dispatch("ESC" if data == b"\x1b" else data.decode(errors="ignore").upper())

    def reset(self):
        # Keep held-state so an autorepeat cannot become a fresh command after reset.
        self.poll()

    def close(self):
        if self.fd is not None:
            if self.terminal_settings is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.terminal_settings)
            else:
                os.close(self.fd)
            self.fd = None
