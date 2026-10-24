"""Standalone Arm6 central pickup and drop, without collection or graph replay.

Run once:
    python -m cloth_agent.center_lift --execute

Uses the same center detection, depth sampling and motion as the collection UI's
reset button: grasp within 80 mm of the detected cloth center, lift 250--300 mm,
randomize yaw within +/-45 degrees, release within 40 mm of home XY, then return
to perception. If the observed folding state is clearly unchanged, the existing
reset policy permits one additional attempt. Results are saved under reports/.

All random_fold_reset options are available, including --max-retries 0 for exactly
one attempt and --image PATH for offline center detection without hardware.
"""


def main(argv=None):
    # Keep the UI and this entry point on the same tested Arm6 reset implementation.
    from .random_fold_reset import main as reset_main
    return reset_main(argv)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('中心抬升已中止，已有记录保留在输出目录。', flush=True)
        raise SystemExit(130)
