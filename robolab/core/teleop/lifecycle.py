"""Separate waiting-for-start input from input for an active recording."""


class AttemptControls:
    def __init__(self):
        self.arm()

    def arm(self):
        self.running = False
        self.start_requested = False
        self.retry_requested = False

    def request_start(self):
        if not self.running:
            self.start_requested = True

    def request_retry(self):
        # There is no active attempt to reject while waiting for N. In
        # particular, do not carry an idle R into the next recording.
        if not self.running:
            return False
        self.retry_requested = True
        return True

    def begin(self):
        if self.running or not self.start_requested:
            raise RuntimeError("An attempt must be armed and started before recording")
        self.running = True
        self.start_requested = False
        self.retry_requested = False


class LeaderControls(AttemptControls):
    """Explicit pause/resume; fresh data alone can never restart an arm."""

    def arm(self):
        super().arm()
        self.paused = None
        self.resume_requested = False
        self.stop_requested = getattr(self, "stop_requested", False)

    def request_start(self):
        if self.running and self.paused:
            self.resume_requested = True
        else:
            super().request_start()

    def pause(self, reason):
        if self.running and not self.paused:
            self.paused = reason
            self.resume_requested = False

    def request_clutch(self):
        if not self.running:
            return
        if self.paused:
            self.resume_requested = True
        else:
            self.pause("clutch")

    def resume(self):
        if not self.running or not self.paused or not self.resume_requested:
            raise RuntimeError("Resume must be explicitly requested while paused")
        self.paused = None
        self.resume_requested = False

    def request_stop(self):
        self.stop_requested = True
