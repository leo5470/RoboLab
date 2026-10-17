"""Native-15 simulation-time checks, independent of Isaac Sim and wall time.

No function in this module resamples a trajectory. The canonical row grid is
returned only after every recorded action interval and boundary has passed.
"""

from __future__ import annotations

import math

import numpy as np


NATIVE_CONTROL_HZ = 15.0
NATIVE_PHYSICS_DT = 1.0 / 120.0
NATIVE_DECIMATION = 8
TIMESTAMP_TOLERANCE_S = 1e-8


def validate_native15_timeline(
    sim_before,
    sim_after,
    declared_control_hz=None,
    physics_dt=None,
    decimation=None,
    *,
    tolerance_s=TIMESTAMP_TOLERANCE_S,
):
    """Validate a recorded native-15 episode and return a JSON-safe report.

    ``sim_before``/``sim_after`` must retain float64 precision. A global clock
    origin is allowed. An N=1 episode still has one verifiable action interval.
    Tolerance is absolute seconds; relative tolerances would hide gaps on large
    simulation counters. The returned ``timestamps_s`` grid is suitable for
    row/video indexing, but does not replace the raw archived timestamps.
    """
    if not math.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ValueError("Timestamp tolerance must be finite and positive")
    errors = []

    def fail(code, message, **details):
        errors.append({"code": code, "message": message, **details})

    result = {
        "valid": False,
        "structural_valid": False,
        "native_15_hz": False,
        "sample_count": 0,
        "target_hz": NATIVE_CONTROL_HZ,
        "tolerance_s": float(tolerance_s),
        "origin_s": None,
        "timestamps_s": [],
        "recorded_elapsed_s": [],
        "action_duration_s": None,
        "measured_action_duration_s": None,
        "last_row_timestamp_s": None,
        "errors": errors,
    }

    arrays = []
    for name, values in (("sim_before_s", sim_before), ("sim_after_s", sim_after)):
        if values is None:
            fail("missing_sim_timestamp", f"Missing required timing/{name}", field=name)
            continue
        arr = np.asarray(values)
        if arr.dtype.kind != "f" or arr.dtype.itemsize != 8:
            fail("timestamp_precision", f"timing/{name} must retain float64 precision", field=name)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            fail("timestamp_shape", f"timing/{name} must be a one-dimensional sample vector", field=name)
            continue
        try:
            arr = arr.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            fail("timestamp_type", f"timing/{name} is not numerical", field=name)
            continue
        if not np.all(np.isfinite(arr)):
            fail("nonfinite_sim_timestamp", f"timing/{name} contains nonfinite values", field=name)
        arrays.append(arr)

    for name, value, expected, tol in (
        ("control_hz", declared_control_hz, NATIVE_CONTROL_HZ, 1e-9),
        ("physics_dt", physics_dt, NATIVE_PHYSICS_DT, tolerance_s),
        ("decimation", decimation, NATIVE_DECIMATION, 0),
    ):
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if isinstance(value, bool) or not math.isfinite(numeric) or abs(numeric - expected) > tol:
            fail("non_native_configuration", f"Declared {name}={value!r} is incompatible with native 15 Hz", field=name)

    if len(arrays) != 2:
        return result
    before, after = arrays
    result["sample_count"] = int(len(before))
    if len(before) != len(after):
        fail("timestamp_count", "Before/after timing counts differ", before_count=len(before), after_count=len(after))
        return result
    if not len(before):
        fail("empty_timeline", "A native-15 episode must contain at least one action interval")
        return result
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        return result
    expected_dt = 1.0 / NATIVE_CONTROL_HZ
    intervals = after - before
    result["origin_s"] = float(before[0])
    result["recorded_elapsed_s"] = (before - before[0]).tolist()
    result["measured_action_duration_s"] = float(after[-1] - before[0])
    result["interval_min_s"] = float(intervals.min())
    result["interval_max_s"] = float(intervals.max())
    result["max_interval_error_s"] = float(np.max(np.abs(intervals - expected_dt)))
    bad = np.flatnonzero(np.abs(intervals - expected_dt) > tolerance_s)
    if bad.size:
        fail("non_native_interval", "Recorded action durations are not 1/15 simulated second", indices=bad.tolist())
    if np.any(intervals <= 0):
        fail("nonpositive_interval", "Simulation after-time must be later than before-time")
    if len(before) > 1:
        continuity = before[1:] - after[:-1]
        bad = np.flatnonzero(np.abs(continuity) > tolerance_s)
        if bad.size:
            fail("discontinuous_simulation", "Next action before-time does not equal preceding after-time", indices=(bad + 1).tolist())
        bad = np.flatnonzero(np.abs(np.diff(before) - expected_dt) > tolerance_s)
        if bad.size:
            fail("non_native_spacing", "Consecutive actions are not spaced at 1/15 simulated second", indices=(bad + 1).tolist())
    canonical = np.arange(len(before), dtype=np.float64) / NATIVE_CONTROL_HZ
    bad = np.flatnonzero(np.abs((before - before[0]) - canonical) > tolerance_s)
    if bad.size:
        fail("non_native_grid", "Recorded elapsed times differ from the episode-local native-15 grid", indices=bad.tolist())
    cadence_errors = {"non_native_configuration", "non_native_interval", "non_native_spacing", "non_native_grid"}
    result["structural_valid"] = not any(error["code"] not in cadence_errors for error in errors)
    if not errors:
        result.update(
            valid=True,
            native_15_hz=True,
            timestamps_s=canonical.tolist(),
            action_duration_s=float(len(before) / NATIVE_CONTROL_HZ),
            last_row_timestamp_s=float(canonical[-1]),
        )
    return result
