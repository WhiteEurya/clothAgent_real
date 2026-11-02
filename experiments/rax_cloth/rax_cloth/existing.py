"""Host-only thin adapters. Never copy these or the production package into agent environments."""
from __future__ import annotations


class ExistingGeometry:
    def __init__(self, config_path, repository):
        from cloth_agent.dual_arm.config import DualConfig
        # Read-only geometry does not require a predefined grasp orientation.
        # Existing loader's homing_only option permits null orientation without
        # relaxing geometry checks. This object never authorizes execution.
        self.config = DualConfig.load(config_path, root=repository, homing_only=True)
        actual = {a.axis: k for k, a in self.config.arms.items()}
        if actual != {6: 'left', 7: 'right'}:
            raise ValueError('physical arm assignment differs from audited mapping')

    def base_pose(self, arm, pose_mm_deg):
        from .backend import ARM_MAP
        return self.config.arms[ARM_MAP[arm]['internal']].base_pose(pose_mm_deg).tolist()

    def joint_box(self, models, lower_deg, upper_deg, *, tool_contact=False):
        from cloth_agent.dual_arm.stop_sweep import verify_joint_box
        return verify_joint_box(self.config, models, lower_deg, upper_deg, tool_contact=tool_contact)

    @staticmethod
    def require_safe(result):
        from cloth_agent.dual_arm.stop_sweep import require_safe
        return require_safe(result)


class ExistingPlanner:
    """Use an operator-created existing DualArmPlanner; no changes to planner options."""
    def __init__(self, planner):
        self.planner = planner

    def joints(self, start_rad, goal_rad):
        return self.planner.paths.joint_path(start_rad, goal_rad)

    def cartesian(self, start_rad, world_targets_m):
        return self.planner.paths.cartesian(start_rad, world_targets_m)

    def ik(self, **arguments):
        return self.planner.solve_grasp_pose(**arguments)


class ProtectedControllerPort:
    """No usable real transport until the existing controller service is commissioned.

    Do not instantiate XArmConnection or bypass DualArmCoordinator here. Its
    watchdog/lease integration must first be tested by the operator-owned service.
    This hard gate cannot be changed through MCP or an agent mode argument.
    """
    def __init__(self):
        raise PermissionError('Real transport disabled: protected service integration pending')
