"""A GPU-rendered operator inset, independent of recorded camera sensors.

Imports of Kit/USD are deliberately lazy so layout validation needs no simulator.
"""


def inset_resolution(width: int, height: int) -> tuple[int, int]:
    """One third of the stream in each dimension, with even render dimensions."""
    if width < 240 or height < 180 or width % 2 or height % 2:
        raise ValueError("the inset layout needs even stream dimensions of at least 240x180")
    return max(64, width // 6 * 2), max(48, height // 6 * 2)


def pin_viewport_resolution(resolution):
    """Override Kit's saved viewport texture size after environment creation."""
    from omni.kit.viewport.utility import get_active_viewport_window

    window = get_active_viewport_window()
    if window is None:
        raise RuntimeError("No operator viewport is available")
    # Kit can restore a saved 1280x720 texture independently of app window size
    # and env.cfg.viewer.resolution. Pin both widget and backing texture so UI
    # layout/resize cannot silently expand them again.
    window.viewport_widget.expand_viewport = False
    window.viewport_widget.fill_frame = False
    window.viewport_api.fill_frame = False
    window.viewport_widget.resolution = tuple(resolution)
    window.viewport_api.resolution_scale = 1.0
    return window.viewport_api.resolution


def wrist_parent_path(camera_prim_path: str, env_prim_path: str) -> str:
    """Resolve a scene camera's mount without using its policy sensor prim."""
    path = camera_prim_path.replace("{ENV_REGEX_NS}", env_prim_path)
    if not path.startswith("/") or "{" in path or path.count("/") < 2:
        raise ValueError(f"Cannot resolve wrist camera mount: {path}")
    return path.rsplit("/", 1)[0]


class OperatorInset:
    """Embed a wrist-mounted (default) or fixed top-down operator view.

    No image readback, image sensor, second stream, or per-step Python work is
    added. Kit composites the inset into the application's existing framebuffer.
    The overview retains mouse navigation; the inset follows its camera mount.
    """

    def __init__(self, width: int, height: int, target, *, camera="wrist",
                 wrist_camera_cfg=None, env_prim_path="/World/envs/env_0"):
        import carb.settings
        import omni.ui as ui
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport_window
        from omni.kit.widget.viewport import ViewportWidget
        from pxr import Gf, Usd, UsdGeom

        if camera not in ("wrist", "top"):
            raise ValueError(f"Unknown inset camera: {camera}")
        self.camera = camera
        self.resolution = inset_resolution(width, height)
        self._widget = None
        self._frame = None
        self._camera_path = None
        self._stage = None
        self.visible = False
        window = get_active_viewport_window()
        if window is None:
            raise RuntimeError("The inset needs a displayed viewport; use --livestream 2 or a local GUI")
        self._stage = omni.usd.get_context(window.viewport_api.usd_context_name).get_stage()
        if self._stage is None:
            raise RuntimeError("Create the environment before the operator inset")

        try:
            if self.camera == "wrist":
                if wrist_camera_cfg is None:
                    raise ValueError("The wrist inset needs the robot's calibrated camera configuration")
                if wrist_camera_cfg.offset.convention != "opengl":
                    raise ValueError("The wrist inset expects an OpenGL-convention camera offset")
                parent = wrist_parent_path(wrist_camera_cfg.prim_path, env_prim_path)
                if not self._stage.GetPrimAtPath(parent):
                    raise RuntimeError(f"Wrist camera mount does not exist: {parent}")
                camera_path = f"{parent}/RoboLabTeleopWrist"
            else:
                camera_path = "/RoboLabTeleopTop"
            # Only create a private session-layer camera. Never edit the baked
            # wrist_camera or spawn the policy wrist_cam sensor/render product.
            self._camera_path = omni.usd.get_stage_next_free_path(self._stage, camera_path, False)
            with Usd.EditContext(self._stage, self._stage.GetSessionLayer()):
                usd_camera = UsdGeom.Camera.Define(self._stage, self._camera_path)
                if self.camera == "wrist":
                    lens = wrist_camera_cfg.spawn
                    usd_camera.CreateFocalLengthAttr(lens.focal_length)
                    usd_camera.CreateHorizontalApertureAttr(lens.horizontal_aperture)
                    usd_camera.CreateVerticalApertureAttr(lens.vertical_aperture)
                    usd_camera.CreateFocusDistanceAttr(lens.focus_distance)
                    usd_camera.CreateClippingRangeAttr(Gf.Vec2f(*lens.clipping_range))
                    usd_camera.AddTranslateOp().Set(Gf.Vec3d(*wrist_camera_cfg.offset.pos))
                    w, x, y, z = wrist_camera_cfg.offset.rot
                    rotation = Gf.Quatd(w, Gf.Vec3d(x, y, z)).GetNormalized()
                    usd_camera.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation)
                else:
                    usd_camera.CreateFocalLengthAttr(18.0)
                    usd_camera.CreateHorizontalApertureAttr(20.955)
                    usd_camera.CreateVerticalApertureAttr(20.955 * self.resolution[1] / self.resolution[0])
                    usd_camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
                    target = Gf.Vec3d(*map(float, target))
                    eye = target + Gf.Vec3d(0, 0, 1.5)
                    pose = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 1, 0)).GetInverse()
                    usd_camera.AddTransformOp().Set(pose)

            # Use an overlay on the existing viewport, not a second floating
            # window (which can cover the stream or render at full resolution).
            self._frame = window.get_frame("robolab.operator_inset")
            self._frame.visible = True
            inset_width, inset_height = self.resolution
            top_margin = 8 if carb.settings.get_settings().get("/app/window/hideUi") else 48
            with self._frame:
                with ui.VStack():
                    ui.Spacer(height=top_margin)
                    with ui.HStack(height=inset_height + 20):
                        ui.Spacer()
                        with ui.VStack(width=inset_width):
                            with ui.ZStack(height=20):
                                ui.Rectangle(style={"background_color": 0xEE171717})
                                ui.Label(f"  {self.camera.upper()} VIEW  |  B: hide/show",
                                         style={"color": 0xFFFFFFFF, "font_size": 13})
                            self._widget = ViewportWidget(
                                usd_context_name=window.viewport_api.usd_context_name,
                                camera_path=self._camera_path,
                                resolution=self.resolution,
                                width=inset_width, height=inset_height,
                            )
                        ui.Spacer(width=8)
                    ui.Spacer()
            self.visible = True
        except Exception:
            self.close()
            raise

    @property
    def viewport_api(self):
        return self._widget.viewport_api if self._widget is not None else None

    def toggle(self):
        self.visible = not self.visible
        self.viewport_api.updates_enabled = self.visible
        self._frame.visible = self.visible
        print(f"{self.camera.title()} inset: {'ON' if self.visible else 'OFF (rendering disabled)'}. "
              "Press B to toggle.", flush=True)

    def close(self):
        if self._widget is not None:
            self._widget.viewport_api.updates_enabled = False
            self._widget.destroy()
            self._widget = None
        if self._frame is not None:
            self._frame.clear()
            self._frame.visible = False
            self._frame = None
        if self._stage is not None and self._camera_path is not None:
            from pxr import Usd

            with Usd.EditContext(self._stage, self._stage.GetSessionLayer()):
                self._stage.RemovePrim(self._camera_path)
            self._camera_path = None
        self._stage = None
        self.visible = False
