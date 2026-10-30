# Install and pin SteamVR 2.16.7 for ALVR

This guide reproduces the Linux SteamVR **2.16.7, build 23791826** setup used
for RoboLab → Isaac Sim/OpenXR → SteamVR → ALVR → USB → Quest 3S.
It installs a separate copy at `~/VR/SteamVR-2.16.7`, outside Steam's managed
library, and routes Steam and ALVR through a dedicated launcher.

The setup was validated on October 22, 2026 with an X11 desktop, NVIDIA GPU,
and **ALVR 20.14.1**. SteamVR reached **Ready**, ALVR streamed near 72 FPS,
and the operator confirmed that the image followed head movement. RoboLab's
XR preflight passed. ALVR still reported roughly 1.2 seconds of latency,
mostly in the game stage; this validates installation and transport, not
end-to-end teleoperation latency. Wayland was not validated.

Steam's `previous` branch is a moving target: it reported 2.17.10 during this
setup. Use the exact depot manifests below to obtain 2.16.7.

## 1. Prerequisites and paths

Use a terminal on the host's local GPU desktop, as the same non-root user who
runs Steam and ALVR. These instructions assume a native Steam installation
at `~/.local/share/Steam`, not Flatpak Steam. Install SteamVR through Steam
and launch it once to provision dependencies. Keep that managed installation.
Install matching ALVR 20.14.1 host and headset clients and configure the wired
connection as described in [Quest setup](quest_viewing.md#host-and-headset-setup).

The commands use Bash, Python 3, `patchelf`, and the usual GNU file utilities.
On Debian/Ubuntu, install `patchelf` if missing:

```bash
sudo apt-get install patchelf
```

Keep the same terminal for the following steps. Adjust these paths for your
installation; the launcher and compositor script below also contain paths
that must match:

```bash
export steam_root="$HOME/.local/share/Steam"
export pinned="$HOME/VR/SteamVR-2.16.7"
export alvr_root="$HOME/.local/share/ALVR-Launcher/installations/v20.14.1/alvr_streamer_linux"
test -x "$steam_root/ubuntu12_32/steam-runtime/run.sh"
test -d "$alvr_root/lib64/alvr"
test -d "$alvr_root/share/vulkan/explicit_layer.d"
```

Stop if any prerequisite check fails. Close SteamVR and ALVR before copying
or patching runtime files. Steam can remain open for the downloads.

## 2. Download the exact depots

Open Steam's console:

```bash
steam steam://open/console
```

Enter each command below **in the Steam console**, waiting for its completion
message before continuing:

```text
download_depot 250820 250823 730365483462960096
download_depot 250820 250824 3814264265268227996
download_depot 250820 250827 9126840328386630080
download_depot 250820 250830 1874458903680892192
download_depot 250820 250832 3112697343985428860
```

These are the five depot/manifest pairs recorded in the working installation's
`pinned-install.json`. The original installation verified 5,785 regular files
against the cached Steam manifests before applying compatibility fixes.

Steam prints the download destination. On the tested host it was
`~/.local/share/Steam/ubuntu12_32/steamapps/content/app_250820`, **not**
`~/.local/share/Steam/steamapps/content/app_250820`. If you already downloaded
all five exact manifests, reuse them. Do not combine depots from other builds.

Merge their contents into a new directory, preserving executable permissions
and symlinks. This block refuses to overwrite an existing pinned installation:

```bash
export depot_root="$steam_root/ubuntu12_32/steamapps/content/app_250820"
python3 - <<'PY'
import os
import shutil
from pathlib import Path

source = Path(os.environ['depot_root'])
target = Path(os.environ['pinned'])
depots = [250823, 250824, 250827, 250830, 250832]
for depot in depots:
    if not (source / f'depot_{depot}').is_dir():
        raise SystemExit(f'Missing depot: {depot}')
if target.exists() or target.is_symlink():
    raise SystemExit(f'Refusing to overwrite {target}')
target.mkdir(parents=True)
for depot in depots:
    shutil.copytree(source / f'depot_{depot}', target,
                    dirs_exist_ok=True, symlinks=True)
print(f'Copied all five depots to {target}')
PY
```

If copying fails, resolve the failure and move the incomplete target aside
before retrying. Keep an unmodified snapshot before continuing:

```bash
tar -czf "$HOME/VR/SteamVR-2.16.7-original-$(date +%Y%m%dT%H%M%S).tar.gz" \
  -C "$HOME/VR" SteamVR-2.16.7
```

## 3. Apply the ALVR compatibility fixes

These are local workarounds validated with ALVR 20.14.1. Apply them to the
pinned copy only, while SteamVR is stopped.

### Preserve Qt lookup during native restarts

Connecting the headset can restart SteamVR. On this setup, that restart lost
the bundled Qt library search path and `vrmonitor` exited because it could not
load `libQt5Multimedia.so.5`. Embed the bundled library path:

```bash
cp -p "$pinned/bin/linux64/vrmonitor" "$pinned/bin/linux64/vrmonitor.original"
patchelf --force-rpath --set-rpath '$ORIGIN/qt/lib:$ORIGIN' \
  "$pinned/bin/linux64/vrmonitor"
patchelf --print-rpath "$pinned/bin/linux64/vrmonitor"
env -u LD_LIBRARY_PATH ldd "$pinned/bin/linux64/vrmonitor"
```

The printed RPATH should be `$ORIGIN/qt/lib:$ORIGIN`; `ldd` should have no
`not found` entries. The single quotes around `$ORIGIN` are required.

### Keep the native compositor filename

ALVR 20.14.1 normally replaces `vrcompositor` with a wrapper symlink and renames
the native executable to `vrcompositor.real`. On this setup SteamVR 2.16.7
failed to recognize the renamed compositor and timed out, despite compositor
startup completing. Keep the executable named `vrcompositor`, and set the
capture environment in `vrcompositor-launcher.sh` instead.

On the fresh depot copy, preserve a native recovery copy:

```bash
python3 - <<'PY'
import os
import shutil
from pathlib import Path

native = Path(os.environ['pinned']) / 'bin/linux64/vrcompositor'
saved = native.with_name('vrcompositor.real')
if native.is_symlink() or not native.is_file() or saved.exists():
    raise SystemExit('Expected a fresh, unwrapped depot compositor; inspect before proceeding')
shutil.copy2(native, saved)
PY
cp -p "$pinned/bin/linux64/vrcompositor-launcher.sh" \
  "$pinned/bin/linux64/vrcompositor-launcher.sh.original"
```

Edit `$pinned/bin/linux64/vrcompositor-launcher.sh`. Insert the following block
**after** its `vrenv.sh` relaunch block and **before** the final
`exec "$ROOT/vrcompositor-launcher" "$@"`. Leave the existing script intact:

```bash
# ALVR capture, while retaining the native vrcompositor filename.
alvr_root="$HOME/.local/share/ALVR-Launcher/installations/v20.14.1/alvr_streamer_linux"
export VK_LAYER_PATH="$alvr_root/share/vulkan/explicit_layer.d"
export VK_INSTANCE_LAYERS=VK_LAYER_ALVR_capture
export DISABLE_VK_LAYER_VALVE_steam_fossilize_1=1
export DISABLE_MANGOHUD=1
export DISABLE_VKBASALT=1
export DISABLE_OBS_VKCAPTURE=1
export VK_LOADER_LAYERS_ENABLE=VK_LAYER_ALVR_capture,VK_LAYER_MESA_device_select
export VK_LOADER_LAYERS_DISABLE='*'
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export LD_PRELOAD="$alvr_root/libexec/alvr/alvr_drm_lease_shim.so"
    export ALVR_SESSION_JSON="$HOME/.config/alvr/session.json"
fi
```

These settings follow the [ALVR 20.14.1 wrapper source](https://github.com/alvr-org/ALVR/blob/v20.14.1/alvr/vrcompositor_wrapper/src/main.rs).
The Wayland branch is retained for parity; this guide's validation covers X11.

### Use the shared host Steam runtime

Run both `vrserver` and `vrcompositor` through the host's Steam **scout**
runtime. ALVR uses a Unix socket at `$XDG_RUNTIME_DIR/alvr-ipc` to pass frames.
Putting the server in a Steam Linux Runtime `sniper`/pressure-vessel container
while the compositor runs on the host hid that socket and produced a blank
image even when the dashboard said **Streaming**.

The launcher in the next section implements the working route through
`ubuntu12_32/steam-runtime/run.sh` and `bin/vrmonitor.sh`. Use that launcher
instead of `vrstartup.sh` or `run-in-sniper` for this setup.

## 4. Install the pinned launcher

First record the patched core files. These checksums detect subsequent changes;
generating them is not independent verification of the downloaded version.
Do not regenerate them merely to suppress a future integrity failure.

```bash
(
  cd "$pinned" || exit 1
  sha256sum \
    bin/linux64/vrserver bin/linux64/vrclient.so bin/linux64/vrmonitor \
    bin/linux64/vrcompositor.real bin/vrmonitor.sh bin/vrpathreg.sh \
    bin/vrenv.sh bin/version.txt steamxr_linux64.json \
    bin/linux64/vrcompositor-launcher.sh > core.sha256
)
mkdir -p "$HOME/.local/bin"
```

Save the following as `~/.local/bin/steamvr-2.16.7`:

```bash
#!/usr/bin/env bash
# Pinned SteamVR 2.16.7 for RoboLab + ALVR. Steam does not manage this directory.
set -euo pipefail
# Steam appends its original launch command after this marker. Discard it so
# binaries from the managed SteamVR installation cannot be launched by accident.
if [[ "${1:-}" == --from-steam ]]; then
    set --
fi
# Steam/ALVR may pass environment variables belonging to another SteamVR copy.
unset STEAMVR_VRENV STEAMVR_TOOLSDIR VRCOMPOSITOR_LD_LIBRARY_PATH
pinned="$HOME/VR/SteamVR-2.16.7"
steam_root="$HOME/.local/share/Steam"
alvr_root="$HOME/.local/share/ALVR-Launcher/installations/v20.14.1/alvr_streamer_linux"
scout="$steam_root/ubuntu12_32/steam-runtime/run.sh"
cd "$pinned"
sha256sum --check --status core.sha256 || {
    echo 'Pinned SteamVR files changed; restore the backup before launching.' >&2
    exit 1
}
# ALVR's Launch SteamVR button reinstalls its wrapper symlink. Restore the
# native executable after checking its trusted .real copy, on every launch.
if [[ -L "$pinned/bin/linux64/vrcompositor" ]]; then
    rm -- "$pinned/bin/linux64/vrcompositor"
    cp -p -- "$pinned/bin/linux64/vrcompositor.real" "$pinned/bin/linux64/vrcompositor"
fi
cmp -s -- "$pinned/bin/linux64/vrcompositor.real" "$pinned/bin/linux64/vrcompositor" || {
    echo 'Pinned compositor differs from its verified native copy.' >&2
    exit 1
}
export XR_RUNTIME_JSON="$pinned/steamxr_linux64.json"
export STEAM_BASE_FOLDER="$steam_root"
if [[ "${1:-}" == --check ]]; then
    printf 'SteamVR 2.16.7 (build 23791826): core checksums OK\nRuntime: %s\n' "$pinned"
    exec "$pinned/bin/vrpathreg.sh" show
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo 'Launch this from the local GPU desktop session.' >&2
    exit 1
fi
if ! [[ -x "$scout" ]]; then
    echo "Steam host runtime is missing: $scout" >&2
    exit 1
fi
# Register again in case launching Steam's managed SteamVR changed the registry.
"$pinned/bin/vrpathreg.sh" setthis
# ALVR capture uses a Unix socket in XDG_RUNTIME_DIR. Keep vrserver and
# vrcompositor on the host together; a private pressure-vessel runtime directory
# would hide vrserver's alvr-ipc socket from the host compositor.
if [[ -n "${PRESSURE_VESSEL_RUNTIME:-}" ]]; then
    echo 'Launch pinned SteamVR from the local host desktop, outside a runtime container.' >&2
    exit 1
fi
exec "$scout" "$pinned/bin/vrmonitor.sh" "$@"
```

Then make it executable and include a copy in the installation for recovery:

```bash
chmod 755 "$HOME/.local/bin/steamvr-2.16.7"
cp -p "$HOME/.local/bin/steamvr-2.16.7" "$pinned/launch-pinned.sh"
```

ALVR's launch button can reinstall its compositor wrapper symlink. The
launcher checks the saved native executable and restores the original filename
before every launch, including `--check`.

## 5. Select the runtime and route launch buttons

Back up existing user settings before changing defaults:

```bash
export settings_backup="$HOME/VR/steamvr-settings-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$settings_backup"
for relative in openvr/openvrpaths.vrpath openxr/1/active_runtime.json; do
  source_path="$HOME/.config/$relative"
  if [ -e "$source_path" ] || [ -L "$source_path" ]; then
    cp -a "$source_path" "$settings_backup/$(basename "$relative")"
  else
    touch "$settings_backup/$(basename "$relative").was-absent"
  fi
done
printf 'Settings backup: %s\n' "$settings_backup"
```

Register OpenVR, preserve/add the ALVR external driver, and set the user's
default OpenXR runtime:

```bash
"$pinned/bin/vrpathreg.sh" setthis
"$pinned/bin/vrpathreg.sh" adddriver "$alvr_root/lib64/alvr"
mkdir -p "$HOME/.config/openxr/1"
ln -sfn "$pinned/steamxr_linux64.json" \
  "$HOME/.config/openxr/1/active_runtime.json"
"$HOME/.local/bin/steamvr-2.16.7" --check
```

The first OpenVR runtime entry should be the pinned directory. An older
managed entry may remain as a fallback. The external driver list should
include the ALVR `lib64/alvr` directory.

In **Steam → Library → SteamVR → Properties → General → Launch Options**,
save the previous value and replace it with the output of:

```bash
printf '"%s/.local/bin/steamvr-2.16.7" --from-steam %%command%%\n' "$HOME"
```

For example, this host uses:

```text
"/home/leocheng/.local/bin/steamvr-2.16.7" --from-steam %command%
```

Use your absolute home path in the UI. The launcher discards Steam's appended
original command after `--from-steam`, so it cannot accidentally execute the
managed SteamVR binary. ALVR's **Launch SteamVR** button starts Steam app
250820 and therefore uses the same launch option. Changing OpenXR's default
alone does not change what ALVR's button launches.

Prefer the Steam UI over editing `localconfig.vdf` while Steam is running;
Steam can overwrite live file edits. If editing that file manually, fully
exit Steam and back it up first.

## 6. Verify the installation

Start the ALVR dashboard and leave the connected headset's ALVR client open.
From the local desktop terminal, run:

```bash
"$HOME/.local/bin/steamvr-2.16.7"
```

Check all of the following:

1. SteamVR reaches **Ready**, ALVR reports **Streaming**, and the headset
   displays an image that responds to head rotation and translation.
2. The latest startup entries in `vrmonitor.txt`, `vrserver.txt`, and
   `vrcompositor.txt` under `~/.local/share/Steam/logs` report **2.16.7**.
   Logs include old sessions; stale notices mentioning 2.18.4 are not proof
   that the current runtime is 2.18.4. `bin/version.txt` contains a timestamp,
   and executing `vrserver --version` starts a server rather than safely
   querying a version. Use startup logs instead.
3. The running executables resolve into `~/VR/SteamVR-2.16.7`:

   ```bash
   for process in vrmonitor vrserver vrcompositor; do
     for pid in $(pgrep -u "$(id -u)" -x "$process"); do
       readlink -f "/proc/$pid/exe"
     done
   done
   ```

4. After a clean SteamVR exit, test Steam's **Play** button and ALVR's
   **Launch SteamVR** button separately and confirm the same version/path.
5. From the RoboLab repository root, run the preflight:

   ```bash
   .venv/bin/python examples/view_xr.py --check \
     --runtime-json "$HOME/VR/SteamVR-2.16.7/steamxr_linux64.json"
   ```

   `errors: []` is a passing preflight. A generic headset-connectivity warning
   can remain even while ALVR is streaming. Preflight does not prove that
   frames reach the headset; the physical headset check is also required.

Continue with [RoboLab XR viewing and teleoperation](quest_viewing.md#preflight-and-launch).
SteamVR Home is optional for the transport check. If `steamtours` opens a
fullscreen desktop window, close SteamVR Home and leave SteamVR/ALVR running.

## 7. Prevent updates and preserve recovery files

The pinned directory is outside `steamapps/common`, so Steam's normal app
updates do not replace its files. Steam may still update its managed SteamVR
copy; the launch option routes around that copy. Steam's **Only update this
game when I launch it** setting alone is not a version lock.

The host Steam scout runtime and ALVR installation remain shared dependencies.
Keep ALVR server/client versions matched, and revalidate after changing ALVR,
Steam, or GPU drivers. Do not delete the ALVR installation referenced by the
compositor script. If launch options or runtime defaults are reset, repeat
section 5 and verify the actual running processes.

Once validation passes, close SteamVR and archive the patched installation:

```bash
tar --exclude='*.log' -czf \
  "$HOME/VR/SteamVR-2.16.7-working-$(date +%Y%m%dT%H%M%S).tar.gz" \
  -C "$HOME/VR" SteamVR-2.16.7
```

Keep the original snapshot, patched snapshot, settings backup, and a copy of
the Steam launch option. On the configured host, the existing archives are:

- `~/VR/SteamVR-2.16.7-backup.tar.gz`: original depot snapshot before fixes.
- `~/VR/SteamVR-2.16.7-working-backup.tar.gz`: patched, validated setup.

To restore the existing working archive, stop SteamVR/ALVR, move the current
directory aside, and extract into `~/VR`:

```bash
mv "$HOME/VR/SteamVR-2.16.7" \
  "$HOME/VR/SteamVR-2.16.7-before-restore-$(date +%Y%m%dT%H%M%S)"
tar -xzf "$HOME/VR/SteamVR-2.16.7-working-backup.tar.gz" -C "$HOME/VR"
install -m 755 "$HOME/VR/SteamVR-2.16.7/launch-pinned.sh" \
  "$HOME/.local/bin/steamvr-2.16.7"
```

For a newly created archive, substitute its timestamped filename. Repeat
sections 5–6: Steam launch options, OpenVR/OpenXR defaults, and the external
ALVR installation are not restored by extracting the archive.

To return to Steam-managed SteamVR, stop VR processes, restore the prior Steam
launch option and the saved OpenVR/OpenXR settings. If OpenXR was originally
absent, remove only the symlink created by this guide. Leave the pinned copy
available until the replacement setup is verified.

## Troubleshooting

| Symptom | Check or fix |
| --- | --- |
| `previous` installs 2.17.10 or another version | Use all five exact depot manifests in section 2. |
| ALVR starts 2.18.4 | Verify Steam's per-user launch option, the `--from-steam` argument, and `/proc/<pid>/exe`. Close old VR processes before retesting. |
| SteamVR exits as the headset connects | Inspect the latest monitor log for missing Qt libraries. Verify the patched RPATH and clean `ldd` output. |
| Compositor timeout, error 303 or -203 | Check that the running native executable is named `vrcompositor`, not `vrcompositor.real`, and that the ALVR environment block is present. These error codes can also have other causes. |
| Streaming status but blank/frozen image | Verify both server and compositor use the host runtime and can see the same `$XDG_RUNTIME_DIR/alvr-ipc` socket. Check video packet/frame statistics; Streaming status alone is insufficient. |
| ALVR driver disabled after a crash | In SteamVR's add-on management, re-enable ALVR and restart after resolving the crash. |
| Launcher checksum failure | Inspect the changed files and restore the known working archive. Do not blindly generate new hashes. |
| `steamtours` fills the desktop | Close the SteamVR Home application. It is separate from the SteamVR monitor/server/compositor. |
| Image works but response is slow | Inspect ALVR game/render, encode, network, and decode latency separately. The original installation still had high game-stage latency. |
