import numpy as np
import pytest
from cloth_agent.dual_arm.static_arm import StaticArmObstacle
from cloth_agent.dual_arm.geometry import DualArmError
from tests.test_dual_arm_runtime import scene


def test_arm7_connects_only_right_and_uses_explicit_static_geometry(scene,monkeypatch,tmp_path):
    from cloth_agent.dual_arm import cli
    from cloth_agent.dual_arm.execution import DualArmCoordinator
    record=dict(arm='left',joint_angles_deg=scene.config.arms['left'].home_joints.tolist(),
                captured_at='saved time',source_observation='saved observation')
    scene.config.static_arm_obstacles={'left':record}
    opened=[]
    class RealStub:
        simulated=False
        def __init__(self,config,cancel,**kwargs):
            assert config is scene.config.arms['right']
            opened.append(config);self.config=config
        def snapshot(self):
            import time
            q=self.config.home_joints
            return dict(joints=q.tolist(),pose=scene.models['right'].forward(q).tolist(),sampled_monotonic_s=time.monotonic())
        def stop(self):pass
        def disconnect(self):pass
    monkeypatch.setattr(cli,'XArmConnection',RealStub)
    connections=cli.connect(scene.config,scene.models,scene.cancel,True)
    coordinator=DualArmCoordinator(scene.config,scene.models,connections,scene.cancel,tmp_path,realtime=True)
    try:
        assert len(opened)==1 and isinstance(connections['left'],StaticArmObstacle)
        state=coordinator.snapshots()
        assert state['left']['live_feedback'] is False
        assert state['left']['state_source']=='assumed_static_obstacle'
        assert np.allclose(state['left']['joints'],record['joint_angles_deg'])
        assert np.allclose(state['left']['pose'],scene.models['left'].forward(record['joint_angles_deg']))
        assert state['left']['static_obstacle_reference']['captured_at']=='saved time'
        for name in ['prepare','prepare_position','gripper','move_joint_position','move_joint_queue','servo','inverse']:
            with pytest.raises(DualArmError,match='cannot receive'):
                getattr(connections['left'],name)()
        assert connections['left'].stop()['status']=='NOT_CONNECTED'
    finally:coordinator.close()
