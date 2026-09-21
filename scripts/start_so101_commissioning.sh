#!/usr/bin/env bash
# RoboLab server commissioning: USB leader, terminal episode controls, no recording.
set -euo pipefail

robolab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$robolab_root"
export OMNI_KIT_ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1

exec "$robolab_root/.venv/bin/python" "$robolab_root/examples/collect_so101_demos.py" \
  --task BananaInBowlTask \
  --leader-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68010085-if00 \
  --leader-id robolab_so101_leader \
  --leader-python "$robolab_root/.venv-so101/bin/python" \
  --leader-urdf "$robolab_root/output/so101_setup/so101_new_calib.urdf" \
  --teleop-config "$robolab_root/configs/teleop/so101_droid.yaml" \
  --control-source terminal \
  --orientation-mode fixed --no-record \
  --device cpu --livestream 2 --defer-images \
  --view-layout inset --inset-camera wrist \
  --output-dir "$robolab_root/output/so101_commissioning/banana" \
  "$@"
