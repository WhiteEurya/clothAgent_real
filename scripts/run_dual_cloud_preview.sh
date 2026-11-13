#!/usr/bin/env bash
# Camera-only preview. Uses the TensorRT libraries bundled with the local SDK installer.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUTF8=1
cloud_python="${CLOTH_PYTHON:-/home/CNS2026330003/miniconda3/envs/cloth-real/bin/python}"
trt_library_dir="${ZED_TENSORRT_LIB_DIR:-$PWD/../zed-mini-setup/sdk-inspect/dependencies/ai_dependencies/lib}"
if [[ -r "$trt_library_dir/libnvinfer.so.10" ]]; then
    export LD_LIBRARY_PATH="$trt_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
printf -v camera_command '%q ' env "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "$cloud_python" -u scripts/preview_dual_camera_cloud.py "$@"
exec sg zed -c "$camera_command"
