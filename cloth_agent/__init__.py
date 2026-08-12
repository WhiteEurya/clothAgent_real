"""Minimal, safety-gated xArm experimentation loop.

The package deliberately exposes the robot to generated experiment code through
only four operations: ``move``, ``open_gripper``, ``close_gripper`` and ``home``.
"""

from .config import ExperimentConfig, RobotConfig
from .robot_api import RobotAPI, RobotExecutionError, SafetyError
from .perception import ClothCenterPerception, PerceptionConfig, PerceptionError
from .session import AgentSession

__all__ = [
    "AgentSession",
    "ExperimentConfig",
    "ClothCenterPerception",
    "PerceptionConfig",
    "PerceptionError",
    "RobotAPI",
    "RobotConfig",
    "RobotExecutionError",
    "SafetyError",
]
