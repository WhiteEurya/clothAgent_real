#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/CNS2026330003/miniconda3/envs/cali/bin/python}"
REFRESH_S="${REFRESH_S:-0.5}"

usage() {
  echo "Usage: $0 [--dry-run|--real] [recovery/Viser options] [camera overrides]"
  echo
  echo "  --dry-run  Run one iteration without physical robot motion (default)."
  echo "  --real     Run continuous physical execution until interrupted."
  echo
  echo "Recovery (enabled by default for --real):"
  echo "  --recover / --continue-on-recoverable-errors"
  echo "  --no-recover"
  echo "  --max-consecutive-recoverable-failures N"
  echo "  --recovery-backoff-s SECONDS"
  echo
  echo "Read-only Viser (enabled by default for --real):"
  echo "  --viser / --no-viser"
  echo "  --viser-host HOST       (loopback only; default 127.0.0.1)"
  echo "  --viser-port PORT       (default 8765)"
  echo "  --viser-refresh-s SEC   (default 1.0)"
  echo
  echo "Camera overrides for this launch only:"
  echo "  --camera-a-exposure VALUE"
  echo "  --camera-b-exposure VALUE"
  echo "  --camera-a-white-balance VALUE"
  echo "  --camera-b-white-balance VALUE"
  echo
  echo "Optional environment variables: RUN_ID, PYTHON, REFRESH_S, VISER_PORT"
}

mode="--dry-run"
mode_seen=false
camera_args=()
recovery_mode="auto"
max_recoverable_failures="3"
recovery_backoff_s="2"
viser_mode="auto"
viser_host="127.0.0.1"
viser_port="${VISER_PORT:-8765}"
viser_refresh_s="1.0"
while (( $# > 0 )); do
  case "$1" in
    --dry-run|--real)
      if [[ "$mode_seen" == true ]]; then
        echo "Only one of --dry-run or --real may be supplied." >&2
        exit 2
      fi
      mode="$1"
      mode_seen=true
      shift
      ;;
    --camera-a-exposure|--camera-b-exposure|--camera-a-white-balance|--camera-b-white-balance)
      if (( $# < 2 )); then
        echo "$1 requires a numeric value." >&2
        exit 2
      fi
      camera_args+=("$1" "$2")
      shift 2
      ;;
    --recover|--continue-on-recoverable-errors)
      recovery_mode="enabled"
      shift
      ;;
    --no-recover)
      recovery_mode="disabled"
      shift
      ;;
    --max-consecutive-recoverable-failures)
      if (( $# < 2 )); then
        echo "$1 requires an integer value." >&2
        exit 2
      fi
      max_recoverable_failures="$2"
      shift 2
      ;;
    --recovery-backoff-s)
      if (( $# < 2 )); then
        echo "$1 requires a numeric value." >&2
        exit 2
      fi
      recovery_backoff_s="$2"
      shift 2
      ;;
    --viser)
      viser_mode="enabled"
      shift
      ;;
    --no-viser)
      viser_mode="disabled"
      shift
      ;;
    --viser-host)
      if (( $# < 2 )); then
        echo "$1 requires a host value." >&2
        exit 2
      fi
      viser_host="$2"
      shift 2
      ;;
    --viser-port)
      if (( $# < 2 )); then
        echo "$1 requires an integer value." >&2
        exit 2
      fi
      viser_port="$2"
      shift 2
      ;;
    --viser-refresh-s)
      if (( $# < 2 )); then
        echo "$1 requires a numeric value." >&2
        exit 2
      fi
      viser_refresh_s="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

case "$mode" in
  --dry-run)
    run_prefix="claude_global_cli_dry"
    max_iterations=1
    real_args=()
    ;;
  --real)
    run_prefix="claude_global_cli_real"
    max_iterations=0
    real_args=(--enable-real)
    ;;
esac

if [[ "$recovery_mode" == "auto" ]]; then
  if [[ "$mode" == "--real" ]]; then
    recovery_mode="enabled"
  else
    recovery_mode="disabled"
  fi
fi
if [[ "$viser_mode" == "auto" ]]; then
  if [[ "$mode" == "--real" ]]; then
    viser_mode="enabled"
  else
    viser_mode="disabled"
  fi
fi

recovery_args=()
if [[ "$recovery_mode" == "enabled" ]]; then
  recovery_args=(
    --continue-on-recoverable-errors
    --max-consecutive-recoverable-failures "$max_recoverable_failures"
    --recovery-backoff-s "$recovery_backoff_s"
  )
fi

RUN_ID="${RUN_ID:-${run_prefix}_$(date +%Y%m%d_%H%M%S)}"
viewer_pid=""
monitor_pid=""
viser_pid=""

stop_viewer() {
  if [[ -n "$viewer_pid" ]]; then
    kill "$viewer_pid" 2>/dev/null || true
    wait "$viewer_pid" 2>/dev/null || true
    viewer_pid=""
    echo "Dashboard paused while a new perception/planning iteration starts."
  fi
}

start_viewer() {
  if [[ "$viser_mode" == "enabled" ]]; then
    return
  fi
  if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
    return
  fi
  echo "Planning artifacts are ready; starting the live artifact dashboard."
  QT_OPENGL=software \
  QT_XCB_GL_INTEGRATION=none \
  LIBGL_ALWAYS_SOFTWARE=1 \
    "$PYTHON" -m cloth_agent.molmo_artifact_viewer \
      "runs/$RUN_ID" \
      --refresh-s "$REFRESH_S" &
  viewer_pid=$!
}

stop_viser() {
  if [[ -n "$viser_pid" ]]; then
    kill "$viser_pid" 2>/dev/null || true
    wait "$viser_pid" 2>/dev/null || true
    viser_pid=""
  fi
}

start_viser() {
  if [[ "$viser_mode" != "enabled" ]]; then
    return
  fi
  if [[ -n "$viser_pid" ]] && kill -0 "$viser_pid" 2>/dev/null; then
    return
  fi
  echo "Read-only Viser will follow saved perception artifacts at http://$viser_host:$viser_port"
  # Keep the log outside the not-yet-created run directory.  The Viser process
  # itself waits for the CLI to create runs/$RUN_ID and never touches hardware.
  viser_log="$PROJECT_ROOT/runs/${RUN_ID}.viser.log"
  "$PYTHON" -m cloth_agent.molmo_artifact_viser \
    "runs/$RUN_ID" \
    --host "$viser_host" \
    --port "$viser_port" \
    --refresh-s "$viser_refresh_s" \
    >"$viser_log" 2>&1 &
  viser_pid=$!
  sleep 0.5
  if ! kill -0 "$viser_pid" 2>/dev/null; then
    echo "Warning: read-only Viser failed to start; inspect $viser_log" >&2
    viser_pid=""
  fi
}

cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  stop_viser
}

monitor_dashboard() {
  local events_file=""
  local events_line_count=0
  local current_line_count
  local event_line

  viewer_pid=""
  trap stop_viewer EXIT
  trap 'exit 0' INT TERM

  while true; do
    if [[ -z "$events_file" ]]; then
      events_file="$(find "runs/$RUN_ID/results/molmo_keypoint_cli" \
        -type f -name events.jsonl -print -quit 2>/dev/null || true)"
    fi
    if [[ -n "$events_file" ]] && [[ -f "$events_file" ]]; then
      current_line_count="$(wc -l < "$events_file")"
      if (( current_line_count > events_line_count )); then
        while IFS= read -r event_line; do
          if [[ "$event_line" == *'"level": "START"'* ]] && \
            [[ "$event_line" == *'"phase": "iteration"'* ]]; then
            stop_viewer
          elif [[ "$event_line" == *'"level": "DONE"'* ]] && \
            { [[ "$event_line" == *'"phase": "molmo"'* ]] || \
              [[ "$event_line" == *'"phase": "global-planning"'* ]]; }; then
            start_viewer
          fi
        done < <(sed -n "$((events_line_count + 1)),${current_line_count}p" "$events_file")
        events_line_count=$current_line_count
      fi
    fi
    sleep 0.2
  done
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "$PROJECT_ROOT"

echo "Run mode: $mode"
echo "Run ID: $RUN_ID"
echo "Artifacts: $PROJECT_ROOT/runs/$RUN_ID"
if [[ "$recovery_mode" == "enabled" ]]; then
  echo "Recovery: enabled (max consecutive failures=$max_recoverable_failures, backoff=${recovery_backoff_s}s)"
else
  echo "Recovery: disabled"
fi
if [[ "$viser_mode" == "enabled" ]]; then
  echo "Viser: enabled (read-only, http://$viser_host:$viser_port)"
  echo "OpenCV artifact dashboard: disabled (Viser is the sole dashboard)"
else
  echo "Viser: disabled"
  echo "OpenCV artifact dashboard: enabled after planning artifacts are ready"
fi

if [[ "$viser_mode" != "enabled" ]]; then
  monitor_dashboard &
  monitor_pid=$!
fi
start_viser

if "$PYTHON" -m cloth_agent.molmo_keypoint_cli \
  --project-root . \
  --run-id "$RUN_ID" \
  --planning-policy claude_global \
  --perception-config config/perception.free_exploration.json \
  "${camera_args[@]}" \
  --confidence-threshold 0.80 \
  --max-iterations "$max_iterations" \
  "${recovery_args[@]}" \
  "${real_args[@]}"; then
  loop_status=0
else
  loop_status=$?
fi
exit "$loop_status"
