"""Small viewing-only adapter over the installed Kit 107 XRCore API.

No hand/controller device is read and no robot action is generated. Importable
without Kit for lifecycle tests; resolve XRCore and Gf only at runtime.
"""


class NativeXRView:
    def __init__(self, core, settings, runtime_json, camera_transform, *, resolution_multiplier=None):
        self.core = core
        self.settings = settings
        self.runtime_json = str(runtime_json)
        self.camera_transform = camera_transform
        self.resolution_multiplier = resolution_multiplier
        self._previous = {}
        self.active = False
        self.ever_active = False

    def start(self):
        overrides = {
            "/persistent/xr/system/openxr/runtime": "custom",
            "/persistent/xr/system/openxr/activeRuntimeJSON": self.runtime_json,
            "/persistent/xr/profile/vr/system/display": "OpenXR",
            "/persistent/xr/profile/vr/renderQuality": "performance",
            # The viewer's eye is already expressed in stage-world coordinates.
            # Kit's default VR height adjustment can add the wearer's real eye
            # height again when it finds a surface below the camera anchor.
            "/persistent/xr/profile/vr/adjustForUserHeight": False,
            "/xr/profile/vr/tools/layout": "robolab_view_only",
        }
        if self.resolution_multiplier is not None:
            overrides["/persistent/xr/profile/vr/render/resolutionMultiplier"] = self.resolution_multiplier
        for key, value in overrides.items():
            self._previous[key] = self.settings.get(key)
            self.settings.set(key, value)
        print(f"XR resolution multiplier override: {self.resolution_multiplier if self.resolution_multiplier is not None else 'unchanged'}",
              flush=True)
        # Unknown tools layout deliberately leaves the controller action map
        # empty: no object grabbing, scaling, teleport, or robot commands.
        self.core.request_enable_profile("vr")

    def update(self):
        active = bool(self.core.is_xr_enabled() and self.core.is_xr_display_enabled())
        if active and self.core.get_current_profile_name() != "vr":
            raise RuntimeError("Expected the native VR profile; another XR profile became active")
        if active and not self.active:
            # Kit adjusts the viewing space to the requested camera pose. It
            # continues to own head tracking, eye projections, and stereo views.
            self.core.schedule_set_camera(self.camera_transform)
            self.ever_active = True
        changed = active != self.active
        self.active = active
        return changed

    def close(self):
        self.core.request_disable_profile()
        # Also cancel a pending enable that never became active.
        self.settings.set("/xr/profile/vr/enabled", False)
        for key, previous in self._previous.items():
            if previous is None:
                self.settings.destroy_item(key)
            else:
                self.settings.set(key, previous)
        self._previous.clear()


def create_native_view(args, runtime_json):
    import carb
    from omni.kit.xr.core import XRCore
    from pxr import Gf

    # Gf's look-at matrix is a view matrix; XRCore expects camera-to-world.
    transform = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*args.eye), Gf.Vec3d(*args.lookat), Gf.Vec3d(0, 0, 1)
    ).GetInverse()
    return NativeXRView(XRCore.get_singleton(), carb.settings.get_settings(), runtime_json, transform,
                        resolution_multiplier=args.xr_resolution_multiplier)
