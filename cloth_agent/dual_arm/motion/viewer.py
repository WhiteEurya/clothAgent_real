"""Read-only Viser trajectory playback reusing the existing collision viewer."""

from __future__ import annotations

import time

import numpy as np

from ..collision.viewer import CollisionViewer
from .preflight import validate_artifact
from .trajectory import sample_segment


def serve(planner, plan, port=8767):
    import viser

    validate_artifact(
        plan, planner
    )  # Replays may be old; execution TTL does not apply.
    server = viser.ViserServer(
        host="127.0.0.1", port=port, label="Offline dual-arm trajectory"
    )
    viewer = CollisionViewer(server, planner.scene)
    for slider in viewer.sliders.values():
        slider.disabled = True
    server.gui.add_markdown(
        f"**Offline trajectory; no motor connection**\n\n"
        f"Certified nominal clearance lower bound: {plan['minimum_clearance']:.6f} m\n\n"
        f"Transit order: {plan['execution_order']['transit']}. "
        "Cloth/gripper events are annotations; playback does not verify grasp success."
    )
    play = server.gui.add_checkbox("Play", initial_value=False)
    seek = server.gui.add_slider(
        "Time (s)", min=0, max=plan["duration_s"], step=0.02, initial_value=0
    )
    status = server.gui.add_markdown("")
    boundaries = np.array(
        [s["start_time_s"] + s["duration_s"] for s in plan["segments"]]
    )

    def show(t):
        index = min(
            int(np.searchsorted(boundaries, t, side="left")), len(boundaries) - 1
        )
        segment = plan["segments"][index]
        q = sample_segment(segment, t - segment["start_time_s"])[0]
        with viewer.lock:
            viewer.q = {"left": q[:6], "right": q[6:]}
            viewer.update()
            status.content = f"Phase: {segment['phase']}; time {t:.3f} s"

    @seek.on_update
    def changed(event):
        show(event.target.value)

    show(0)
    print(f"Offline trajectory Viser: http://127.0.0.1:{port}", flush=True)
    previous = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if play.value:
                t = min(plan["duration_s"], seek.value + now - previous)
                seek.value = t
                show(t)
                if t >= plan["duration_s"]:
                    play.value = False
            previous = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
