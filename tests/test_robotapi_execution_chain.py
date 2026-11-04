import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from cloth_agent.config import SafetyError
from cloth_agent.experiment import ExperimentRunner
from cloth_agent.robot_api import RobotAPI, RobotExecutionError, XArmBackend
from tests.test_loop import robot_config


def test_new_chain_does_not_import_native_executor_or_ik_solver():
    code = """
import sys
from cloth_agent.experiments.graph_policy import make_runtime
r = make_runtime('config/dual_arm.zed_dual_xarm7_20261103.json', 0)
assert 'cloth_agent.codex_native_runtime' not in sys.modules
assert 'cloth_agent.dual_arm.stop_sweep' not in sys.modules
assert 'cloth_agent.dual_arm.pyroki_ik' not in sys.modules
assert 'jax' not in sys.modules
assert r.config.execution_mode == 'geometry_only'
r.close()
"""
    subprocess.run([sys.executable, '-c', code], check=True)


def test_real_path_rejection_happens_before_sdk_connection(tmp_path, monkeypatch):
    from cloth_agent.robot_collision import RobotCollision
    from tests.test_robot_collision import scene
    c = scene()
    def rejected(*args, **kwargs):
        raise SafetyError('joint sweep collides')
    monkeypatch.setattr(c, 'plan_action', rejected)
    config = replace(robot_config(), robot_ip='left', collision_scene_path='scene', collision_active_arm='left')
    runner = ExperimentRunner(tmp_path, config, collision_checker=c)
    source = runner.workspace / 'fold.py'
    source.write_text('def run():\n    move(300, 0, 100, 0)\n')
    monkeypatch.setattr('cloth_agent.experiment.validate_controller_trajectory', lambda *a: pytest.fail('connected'))
    monkeypatch.setattr('cloth_agent.experiment.XArmBackend', lambda *a: pytest.fail('connected'))
    with pytest.raises(SafetyError, match='joint sweep collides'):
        runner.run_experiment(source, real=True, confirmed=True)
    assert not json.loads((runner.results_dir / 'fold.json').read_text())['actual_robot_actions']
    with pytest.raises(RobotExecutionError, match='requires its collision checker'):
        XArmBackend(config)


def test_command_failure_requests_stop_and_prevents_more_commands():
    class Backend:
        stopped = 0
        def move(self, *args):
            raise RobotExecutionError('fault 11')
        def stop(self):
            self.stopped += 1
    backend = Backend()
    robot = RobotAPI(robot_config(), backend)
    with pytest.raises(RobotExecutionError, match='fault 11'):
        robot.move(300, 0, 100, 0)
    assert backend.stopped == 1
    with pytest.raises(RobotExecutionError, match='halted'):
        robot.open_gripper()


def test_stop_failure_preserves_original_fault():
    class Backend:
        def move(self, *args):
            raise RobotExecutionError('original fault')
        def stop(self):
            raise RuntimeError('transport lost')
    robot = RobotAPI(robot_config(), Backend())
    with pytest.raises(RobotExecutionError, match='original fault'):
        robot.move(300, 0, 100, 0)
    assert 'transport lost' in robot.action_dicts()[0]['error']
