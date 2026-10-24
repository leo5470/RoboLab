"""Compact desktop collection status, created only after Kit starts."""


def status_lines(*, object_name, layout_index, layout_count, task, successes,
                 target_successes, attempts, completed, skipped, state,
                 reset_key="F9", next_key="F12", total_successes=0, target_total_successes=None):
    """Format status; layout_index is zero-based within the selected sequence."""
    state = str(state).upper()
    reset_action = {
        "RECORDING": "retry layout",
        "ARMED": "cancel arming",
        "WAITING": "confirm READY",
        "READY": "confirm READY",
        "SETTLING": "restart preparation",
        "PREPARATION_FAILED": "retry preparation",
    }.get(state, "reset / READY")
    controls = f"{reset_key}: {reset_action}"
    if next_key is not None:
        controls += f"     {next_key}: next layout"
    return (
        f"{object_name} · Layout {layout_index + 1}/{layout_count}",
        task,
        f"Successes {successes}/{target_successes} · Attempts {attempts} · {state}",
        (f"Total {total_successes}/{target_total_successes} · " if target_total_successes is not None else "")
        + f"Completed {completed} · Skipped {skipped}",
        controls,
    )


class CollectionStatus:
    """Update an unfocused, mouse-transparent Kit status window in place."""

    def __init__(self):
        import omni.ui as ui

        flags = (ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING | ui.WINDOW_FLAGS_NO_MOUSE_INPUTS
                 | ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_COLLAPSE
                 | ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_CLOSE
                 | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS)
        self._window = ui.Window("RoboLab Collection", width=550, height=158,
                                 position_x=18, position_y=42, flags=flags)
        with self._window.frame:
            with ui.VStack(spacing=4):
                self._labels = [ui.Label("", height=20) for _ in range(5)]
        self._lines = None

    def update(self, **values):
        if self._window is None:
            return
        lines = status_lines(**values)
        if lines == self._lines:
            return
        for label, line in zip(self._labels, lines):
            label.text = line
        self._lines = lines

    def close(self):
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self._labels = []
