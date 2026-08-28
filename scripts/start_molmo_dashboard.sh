#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/CNS2026330003/miniconda3/envs/cali/bin/python}"
REFRESH_S="${REFRESH_S:-0.5}"

usage() {
  echo "Usage: $0 [--dry-run|--real] [--no-molmo] [--objective TASK] [planning/recovery/Viser options] [camera overrides]"
  echo
  echo "  --dry-run  Run one iteration without physical robot motion (default)."
  echo "  --real     Run continuous physical execution until interrupted."
  echo "  --objective TASK"
  echo "             Natural-language task sent to Claude on every exploration iteration."
  echo "  --no-molmo  Disable the optional global Molmo annotation pass; keep Claude-global RGB-D exploration and skills."
  echo "  --claude-timeout-s N  Max seconds for one Claude planning call (default: 900s)."
  echo "  --max-replans N       Extra Claude correction calls after pre-execution rejection (default: 1)."
  echo
  echo "Recovery (enabled by default for --real):"
  echo "  --recover / --continue-on-recoverable-errors"
  echo "  --no-recover"
  echo "  --max-consecutive-recoverable-failures N (0 = unlimited pre-execution recovery)"
  echo "  --recovery-backoff-s SECONDS"
  echo
  echo "Camera A/B rollout recording (enabled by default):"
  echo "  --record-rollouts"
  echo "  --no-record-rollouts"
  echo "  --recording-no-native  Keep the cumulative MP4 but omit RealSense .db3 files"
  echo "  --combined-video-speed N  Speed up the cumulative video by N (default: 32x)"
  echo "  --hold-checkpoint-timeout-s N  Max Claude hold-classification time (default: 600s, max: 900s)"
  echo "  --hold-checkpoint-settle-s N   Hold still before fresh A/B capture (default: 0.75s)"
  echo "  Completed rollout segments are appended in order to combined_rollout.mp4; old per-iteration video files are pruned after evaluation."
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
  echo "Optional environment variables: RUN_ID, PYTHON, REFRESH_S, VISER_PORT, CLAUDE_TIMEOUT_S, MAX_REPLANS, HOLD_CHECKPOINT_TIMEOUT_S"
}

mode="--dry-run"
mode_seen=false
objective="Take one planning-mode-appropriate agent-chosen action that makes the current garment as open and spread as safely possible."
camera_args=()
recovery_mode="auto"
# A real overnight dashboard should not terminate merely because several
# consecutive pre-execution perception/planning attempts failed. The CLI uses
# 0 as the explicit unlimited-recovery sentinel; hardware and operator errors
# remain non-recoverable and still stop the run.
max_recoverable_failures="0"
recovery_backoff_s="2"
recording_mode="enabled"
recording_native="enabled"
molmo_mode="enabled"
combined_video_speed="32"
hold_checkpoint_timeout_s="${HOLD_CHECKPOINT_TIMEOUT_S:-600}"
hold_checkpoint_settle_s="0.75"
claude_timeout_s="${CLAUDE_TIMEOUT_S:-900}"
max_replans="${MAX_REPLANS:-1}"
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
    --objective)
      if (( $# < 2 )); then
        echo "$1 requires a task description." >&2
        exit 2
      fi
      objective="$2"
      shift 2
      ;;
    --no-molmo)
      molmo_mode="disabled"
      shift
      ;;
    --combined-video-speed)
      if (( $# < 2 )); then
        echo "$1 requires a positive numeric multiplier." >&2
        exit 2
      fi
      combined_video_speed="$2"
      shift 2
      ;;
    --hold-checkpoint-timeout-s)
      if (( $# < 2 )); then
        echo "$1 requires a timeout in seconds." >&2
        exit 2
      fi
      hold_checkpoint_timeout_s="$2"
      shift 2
      ;;
    --hold-checkpoint-settle-s)
      if (( $# < 2 )); then
        echo "$1 requires a settle interval in seconds." >&2
        exit 2
      fi
      hold_checkpoint_settle_s="$2"
      shift 2
      ;;
    --claude-timeout-s)
      if (( $# < 2 )); then
        echo "$1 requires a timeout in seconds." >&2
        exit 2
      fi
      claude_timeout_s="$2"
      shift 2
      ;;
    --max-replans)
      if (( $# < 2 )); then
        echo "$1 requires an integer value." >&2
        exit 2
      fi
      max_replans="$2"
      shift 2
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
    --record-rollouts)
      recording_mode="enabled"
      shift
      ;;
    --no-record-rollouts)
      recording_mode="disabled"
      shift
      ;;
    --recording-no-native)
      recording_native="disabled"
      shift
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

if [[ "$recording_mode" == "enabled" ]]; then
  recording_args=(--record-rollouts)
  if [[ "$recording_native" == "disabled" ]]; then
    recording_args+=(--recording-no-native)
  fi
else
  recording_args=(--no-record-rollouts)
fi
if [[ "$mode" == "--real" && "$recording_mode" != "enabled" ]]; then
  echo "--real requires rollout recording for the online Camera A/B hold checkpoint." >&2
  exit 2
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
echo "Objective: $objective"
if [[ "$recovery_mode" == "enabled" ]]; then
  if [[ "$max_recoverable_failures" == "0" ]]; then
    recovery_limit_label="unlimited"
  else
    recovery_limit_label="$max_recoverable_failures"
  fi
  echo "Recovery: enabled (max consecutive failures=$recovery_limit_label, backoff=${recovery_backoff_s}s)"
else
  echo "Recovery: disabled"
fi
if [[ "$recording_mode" == "enabled" ]]; then
  if [[ "$recording_native" == "enabled" ]]; then
    echo "Rollout recording: enabled (Camera A/B MP4, depth/composite, timestamps, and native .db3)"
  else
    echo "Rollout recording: enabled (Camera A/B MP4, depth/composite, and timestamps; native .db3 disabled)"
  fi
else
  echo "Rollout recording: disabled"
fi
echo "Combined video speed: ${combined_video_speed}x"
if [[ "$molmo_mode" == "enabled" ]]; then
  echo "Global Molmo annotations: enabled"
else
  echo "Global Molmo annotations: disabled (Claude + RGB-D/reference only)"
fi
echo "Lift checkpoints: 3 rising A/B captures, reverse release, settle=${hold_checkpoint_settle_s}s"
echo "Legacy hold timeout setting: ${hold_checkpoint_timeout_s}s (not used by lift-checkpoint mode)"
echo "Claude planning timeout: ${claude_timeout_s}s (extra replans=${max_replans})"
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

molmo_args=()
if [[ "$molmo_mode" == "enabled" ]]; then
  molmo_args+=(--global-molmo-annotations)
fi

if "$PYTHON" -m cloth_agent.molmo_keypoint_cli \
  --project-root . \
  --run-id "$RUN_ID" \
  --planning-policy claude_global \
  "${molmo_args[@]}" \
  --perception-config config/perception.free_exploration.json \
  "${camera_args[@]}" \
  --confidence-threshold 0.80 \
  --objective "$objective" \
  --combined-video-speed "$combined_video_speed" \
  --claude-timeout-s "$claude_timeout_s" \
  --max-replans "$max_replans" \
  --hold-checkpoint-timeout-s "$hold_checkpoint_timeout_s" \
  --hold-checkpoint-settle-s "$hold_checkpoint_settle_s" \
  --max-iterations "$max_iterations" \
  "${recovery_args[@]}" \
  "${recording_args[@]}" \
  "${real_args[@]}"; then
  loop_status=0
else
  loop_status=$?
fi
exit "$loop_status"
