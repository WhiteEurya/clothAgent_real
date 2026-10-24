"""Resolve a model's image-space jaw opening axis before closing the gripper."""
import numpy as np
from scipy.spatial.transform import Rotation

from .dual_arm.geometry import DualArmError
from .grasp_angle import image_angle_to_rpy, transform_for
from .experiments.one_shot_goal_planner_probe_v1 import save, validate_plan


def resolve(runtime, observation, plan, start, home_rpy, folder):
    # Validate again at the physical preparation boundary, including null rules.
    from PIL import Image
    with Image.open(observation.image) as image:
        validate_plan(plan, image.size, require_grasp_rotation=True)
    angle = plan['grasp_angle_deg']
    selected = np.asarray(home_rpy, dtype=float).copy()
    delta = 0.
    if angle is not None:
        intrinsic = np.asarray(runtime.camera.intrinsics, dtype=float)
        world_camera = transform_for(observation)
        if (intrinsic.shape != (3, 3) or world_camera.shape != (4, 4)
                or not np.isfinite(intrinsic).all() or not np.isfinite(world_camera).all()):
            raise DualArmError('invalid calibration for grasp angle')
        tcp_rotation = Rotation.from_euler('xyz', runtime.config.arms['left'].tcp_offset[3:], degrees=True)
        axis_tcp = tcp_rotation.inv().apply([0., 1., 0.])
        selected, delta = image_angle_to_rpy(angle, plan['grasp_px'], start, home_rpy,
                                            intrinsic, world_camera, axis_tcp)
    save(folder/'grasp_orientation.json', dict(
        image_angle_deg=angle, coordinate_frame='perception before/rgb.png',
        axis='jaw opening axis', home_rpy_world_deg=np.asarray(home_rpy).tolist(),
        selected_rpy_world_deg=selected.tolist(), world_z_delta_deg=delta,
        adjusted_before_closure=angle is not None, rotation_while_holding=False))
    return selected


def same_direction(a, b, tolerance_deg=5.):
    """Image opening axes are equivalent modulo 180 degrees."""
    a, b = a.get('grasp_angle_deg'), b.get('grasp_angle_deg')
    if a is None or b is None:
        return a is None and b is None
    return abs((a-b+90) % 180-90) <= tolerance_deg
