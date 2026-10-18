"""Arm7 configuration for the shared single-arm runtime."""
import json
import numpy as np
from .single_arm_runtime import SingleArmRuntime, wrist_equivalent
from .dual_arm.geometry import DualArmError


class Arm7Runtime(SingleArmRuntime):
    def load_config(self,config_path):
        from .codex_camera_b import with_arm7_camera
        config=with_arm7_camera(super().load_config(config_path))
        # Match arm6's runtime-local policy for the active arm only.
        config.collision_envelopes_disabled_arms=frozenset({'right'})
        parked_file=config.root/'data/robot/arm6_parked_for_arm7.json'
        parked=json.loads(parked_file.read_text())
        if parked.get('arm')!='left':raise DualArmError('arm7 static obstacle must be arm6/left')
        joints=np.asarray(parked['joint_angles_deg'],dtype=float)
        if joints.shape!=(config.arms['left'].axis,) or not np.isfinite(joints).all():
            raise DualArmError('invalid arm6 static obstacle joints')
        parked['reference_file']=str(parked_file)
        config.static_arm_obstacles={'left':parked}
        return config
