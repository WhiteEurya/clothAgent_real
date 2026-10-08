"""Viser review of both official URDFs, attached boxes and FCL witnesses.

This module has no robot SDK or movement connection. All sliders are offline.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry import DualArmError
from .checker import CollisionChecker


def wxyz(matrix):
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.r_[q[3], q[:3]]


class CollisionViewer:
    def __init__(self, server, scene):
        from viser.extras import ViserUrdf

        self.server, self.scene = server, scene
        self.checker = CollisionChecker(scene)
        self.lock = threading.RLock()
        self.q = {k: q.copy() for k, q in scene.initial.items()}
        server.scene.set_up_direction("+z")
        self.robots, self.handles, self.sliders = {}, {}, {}
        self.visual = server.gui.add_checkbox(
            "Show official visual meshes", initial_value=True
        )
        self.collision = server.gui.add_checkbox(
            "Show collision geometry", initial_value=True
        )
        self.status = server.gui.add_markdown(
            "Offline model review; no motor connection."
        )
        self.witness = None
        for key, model in scene.models.items():
            base = np.asarray(scene.raw["arms"][key]["world_from_base_m"])
            # Viser URDF is drawn in its root frame, ArmModel FK in link_base.
            root_to_base = np.asarray(model.urdf.get_transform("link_base"))
            view_base = base @ np.linalg.inv(root_to_base)
            server.scene.add_frame(
                f"/robots/{key}",
                position=view_base[:3, 3],
                wxyz=wxyz(view_base),
                show_axes=False,
            )
            self.robots[key] = ViserUrdf(
                server,
                model.config.urdf,
                root_node_name=f"/robots/{key}",
                load_meshes=True,
                load_collision_meshes=False,
            )
            for index, name in enumerate(model.names):
                slider = server.gui.add_slider(
                    f"{key}/{name} (deg)",
                    min=float(np.degrees(model.lower[index])),
                    max=float(np.degrees(model.upper[index])),
                    step=0.1,
                    initial_value=float(np.degrees(self.q[key][index])),
                )
                self.sliders[key, index] = slider

                @slider.on_update
                def updated(event, arm=key, joint=index):
                    with self.lock:
                        self.q[arm][joint] = np.radians(event.target.value)
                        self.update()

        for shape in scene.shapes:
            color = (60, 160, 245) if shape.arm == "left" else (245, 160, 50)
            if shape.kind == "obstacle":
                color = (130, 130, 130)
            handle = server.scene.add_mesh_simple(
                f"/collision/{shape.name}",
                shape.mesh.vertices,
                shape.mesh.faces,
                color=color,
                opacity=0.5,
            )
            self.handles[shape.name] = handle

        @self.visual.on_update
        def toggle_visual(_):
            for robot in self.robots.values():
                robot.show_visual = self.visual.value

        @self.collision.on_update
        def toggle_collision(_):
            for handle in self.handles.values():
                handle.visible = self.collision.value

        self.update()

    def update(self):
        with self.lock:
            for key, robot in self.robots.items():
                model = self.scene.models[key]
                # Extra URDF drive_joint is for visual gripper articulation;
                # collision uses an explicit envelope covering all openings.
                cfg = np.zeros(len(model.urdf.actuated_joint_names))
                cfg[: model.config.axis] = self.q[key]
                robot.update_cfg(cfg)
            frames = self.scene.transforms(self.q["left"], self.q["right"])
            result = self.checker.check(self.q["left"], self.q["right"])
            highlight = set(result["closest_pair"])
            for shape in self.scene.shapes:
                handle = self.handles[shape.name]
                handle.position = frames[shape.name][:3, 3]
                handle.wxyz = wxyz(frames[shape.name])
                handle.color = (
                    (255, 40, 60)
                    if shape.name in highlight
                    else (
                        (130, 130, 130)
                        if shape.arm is None
                        else (60, 160, 245)
                        if shape.arm == "left"
                        else (245, 160, 50)
                    )
                )
            if self.witness is not None:
                self.witness.remove()
                self.witness = None
            if result["nearest_points_m"] is not None:
                self.witness = self.server.scene.add_line_segments(
                    "/nearest_pair",
                    points=np.asarray([result["nearest_points_m"]]),
                    colors=(255, 40, 60),
                    line_width=5,
                )
            self.status.content = (
                "**OFFLINE — no robot control**\n\n"
                f"Model state: {'CLEAR' if result['safe'] else 'UNSAFE'}; collision={result['collision']}\n\n"
                f"Minimum distance: {result['min_distance']:.6f} m; required residual margin: {result['minimum_margin_m']:.6f} m\n\n"
                f"Closest pair (red): {' ↔ '.join(result['closest_pair'])}\n\n"
                f"Calibration: {result['calibration_status']}; geometry verified: {result['geometry_verified']}\n\n"
                "13 joint sliders are radians internally / degrees in this UI. Gripper boxes must cover all jaw openings. "
                "A clear static model does not certify real motion."
            )
            self.last_result = result
            return result


def serve(scene, host="127.0.0.1", port=8766):
    import viser

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise DualArmError("offline collision viewer binds to loopback only")
    server = viser.ViserServer(
        host=host, port=port, label="Dual-arm collision review (offline)"
    )
    CollisionViewer(server, scene)
    print(f"Offline Viser: http://{host}:{port} (no robot connection)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
