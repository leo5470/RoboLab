"""Keyboard state that tolerates duplicate and out-of-order release events."""

import carb.input
from isaaclab.devices import Se3Keyboard


class HeldKeyState:
    """Track held keys, not accumulated press/release deltas.

    A release after reset is harmless; repeated presses never amplify motion.
    Clearing motion deliberately preserves the gripper latch.
    """

    MOTION_KEYS = frozenset("WSADQEZXTGCV")

    def __init__(self):
        self.held = set()
        self.closed = False

    def reset(self):
        self.held.clear()
        self.closed = False

    def event(self, key, kind):
        if kind == "release":
            self.held.discard(key)
            return False
        if kind != "press" or key in self.held:
            return False
        self.held.add(key)
        if key == "K":
            self.closed = not self.closed
        elif key == "L":
            self.held.difference_update(self.MOTION_KEYS)
        return True


class ResponsiveSe3Keyboard(Se3Keyboard):
    """Keep Isaac's bindings/rotation convention and drain input before sampling."""

    def __init__(self, cfg):
        self.state = HeldKeyState()
        self._command_dirty = True
        self._cached_command = None
        super().__init__(cfg)

    def reset(self):
        super().reset()
        self.state.reset()
        self._command_dirty = True

    def _on_keyboard_event(self, event, *args, **kwargs):
        kinds = {
            carb.input.KeyboardEventType.KEY_PRESS: "press",
            carb.input.KeyboardEventType.KEY_RELEASE: "release",
            carb.input.KeyboardEventType.KEY_REPEAT: "repeat",
        }
        kind = kinds.get(event.type)
        if kind is None:
            return True
        key = event.input.name
        pressed = self.state.event(key, kind)
        if kind != "repeat":
            self._delta_pos.fill(0)
            self._delta_rot.fill(0)
            for held in self.state.held:
                if held in "WSADQE":
                    self._delta_pos += self._INPUT_KEY_MAPPING[held]
                elif held in "ZXTGCV":
                    self._delta_rot += self._INPUT_KEY_MAPPING[held]
            self._close_gripper = self.state.closed
            self._command_dirty = True
        if pressed and key in self._additional_callbacks:
            self._additional_callbacks[key]()
        return True

    def poll(self):
        # WebRTC can enqueue input while physics is running. Distribute it now
        # instead of waiting for the next expensive Kit/RTX update.
        self._input.distribute_buffered_events()

    def advance(self):
        self.poll()
        if self._command_dirty:
            self._cached_command = super().advance()
            self._command_dirty = False
        return self._cached_command

    def close(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            # Carbonite versions differ; Isaac's own destructor uses the old
            # spelling, which does not exist in the installed 5.0 bindings.
            unsubscribe = getattr(self._input, "unsubscribe_from_keyboard_events", None)
            if unsubscribe is None:
                unsubscribe = self._input.unsubscribe_to_keyboard_events
            unsubscribe(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __del__(self):
        self.close()
