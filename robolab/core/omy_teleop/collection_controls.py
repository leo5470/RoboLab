"""Collection hotkeys; validation can run before starting Isaac Sim."""

import re
from collections import Counter


RESERVED_KEYS = {
    "F1": "Help",
    "F2": "Rename",
    "F7": "Toggle UI",
    "F8": "Profiler",
    "F10": "Capture Screenshot",
    "F11": "Fullscreen",
}
SUPPORTED_KEYS = tuple(f"F{index}" for index in range(1, 13) if f"F{index}" not in RESERVED_KEYS)


def validate_keys(reset_key="F9", next_key="F12"):
    """Return normalized keys or raise ValueError, without importing Kit.

    ``next_key=None`` disables layout advancement for single-task collection.
    Known built-in shortcuts are excluded; other installed extensions are checked
    against the running Kit hotkey registry by :class:`CollectionControls`.
    """
    def normalize(value, label):
        if not isinstance(value, str) or not re.fullmatch(r"F(?:[1-9]|1[0-2])", value.strip().upper()):
            raise ValueError(f"{label} must be an unmodified function key: {', '.join(SUPPORTED_KEYS)}")
        key = value.strip().upper()
        if key in RESERVED_KEYS:
            raise ValueError(f"{label} {key} is reserved by Isaac Sim/Kit for {RESERVED_KEYS[key]}")
        return key

    reset_key = normalize(reset_key, "reset key")
    next_key = None if next_key is None else normalize(next_key, "next-layout key")
    if reset_key == next_key:
        raise ValueError("reset and next-layout keys must differ")
    return reset_key, next_key


def _key_name(value):
    return str(getattr(value, "name", value)).upper()


def runtime_key_conflicts(reset_key, next_key):
    """Return conflicting registered Kit actions, or None if unavailable.

    This inspects the public Kit hotkey registry. It cannot discover desktop
    shortcuts or extensions subscribing directly to the Carbonite input stream.
    Local/window-specific bindings also conflict with these global controls.
    """
    try:
        from omni.kit.hotkeys.core import get_hotkey_registry

        registry = get_hotkey_registry()
        if registry is None:
            return None
        hotkeys = registry.get_all_hotkeys()
    except (ImportError, AttributeError, RuntimeError):
        return None

    keys = {reset_key, next_key} - {None}
    conflicts = []
    for hotkey in hotkeys:
        combination = hotkey.key_combination
        if combination is None or combination.modifiers:
            continue
        key = _key_name(combination.key)
        if key in keys:
            conflicts.append(f"{key}: {hotkey.action_text}")
    return tuple(conflicts)


class CollectionControls:
    """Queue one event per unmodified press, preserving held keys on clear.

    Construct after Kit startup and poll from the simulation thread. Modifier
    combinations pass through to Kit and never trigger collection controls.
    A modified press still marks the key held, preventing a later repeat from
    becoming a collection command when its modifier is released first.
    """

    def __init__(self, reset_key="F9", next_key="F12"):
        self.reset_key, self.next_key = validate_keys(reset_key, next_key)
        self._pending = Counter()
        self._held = set()
        self._sub = None
        conflicts = runtime_key_conflicts(self.reset_key, self.next_key)
        if conflicts:
            raise RuntimeError("Collection hotkeys conflict with registered Kit actions: " + "; ".join(conflicts))
        self.binding_check = (
            "Checked registered Kit hotkeys; OS and direct-input bindings are not covered."
            if conflicts is not None else
            "Kit hotkey registry unavailable; only known built-in shortcut exclusions were checked."
        )

        import carb.input
        import omni.appwindow

        self._types = carb.input.KeyboardEventType
        self._modifier_mask = (
            carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT
            | carb.input.KEYBOARD_MODIFIER_FLAG_CONTROL
            | carb.input.KEYBOARD_MODIFIER_FLAG_ALT
            | carb.input.KEYBOARD_MODIFIER_FLAG_SUPER
        )
        self._input = carb.input.acquire_input_interface()
        window = omni.appwindow.get_default_app_window()
        if window is None or window.get_keyboard() is None:
            raise RuntimeError("Collection controls require a Kit window with keyboard input")
        self._keyboard = window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *unused):
        key = _key_name(event.input)
        if key not in (self.reset_key, self.next_key):
            return True
        modified = bool(getattr(event, "modifiers", 0) & self._modifier_mask)
        if event.type == self._types.KEY_RELEASE:
            self._held.discard(key)
        elif event.type == self._types.KEY_PRESS:
            if key not in self._held and not modified:
                self._pending[key] += 1
            self._held.add(key)
        elif event.type != self._types.KEY_REPEAT:
            return True
        # Carbonite's bool return controls propagation to later subscribers.
        # Caps/Num Lock are not shortcut modifiers for function keys.
        return modified

    def _take(self, key):
        if key is not None and self._pending[key]:
            self._pending[key] -= 1
            return True
        return False

    def take(self):
        return self._take(self.reset_key)

    def take_next(self):
        return self._take(self.next_key)

    def clear(self):
        self._pending.clear()

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None
        self.clear()
