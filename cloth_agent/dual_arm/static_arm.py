"""An explicitly assumed stationary obstacle; never opens a controller connection."""
import time
from .geometry import DualArmError, finite


class StaticArmObstacle:
    # The surrounding run is physical. This is obstacle geometry, not a
    # simulated controller allowed to receive motion commands.
    simulated = False
    static_obstacle = True

    def __init__(self, config, model, record):
        self.config=config
        self.record=dict(record)
        self.joints=finite(record['joint_angles_deg'],(config.axis,),'static obstacle joints')
        self.pose=model.forward(self.joints)

    def snapshot(self):
        return dict(joints=self.joints.tolist(),pose=self.pose.tolist(),
                    gripper=dict(state='unobserved',available=False,usable_for_completion=False),
                    sampled_monotonic_s=time.monotonic(),
                    state_source='assumed_static_obstacle',live_feedback=False,
                    static_obstacle_reference=self.record)

    def _reject(self,*args,**kwargs):
        raise DualArmError('static arm obstacle cannot receive controller commands')

    prepare=prepare_position=servo=gripper=move_joint_position=move_joint_queue=inverse=_reject

    def stop(self):
        return dict(status='NOT_CONNECTED',reason='static obstacle; no controller command sent')

    def disconnect(self):
        pass
