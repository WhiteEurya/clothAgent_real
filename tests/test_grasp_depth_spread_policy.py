from pathlib import Path
import numpy as np
import pytest
from cloth_agent.dual_arm.observation import Observation, DepthUnavailableError
from cloth_agent.dual_arm.geometry import DualArmError


def observation():
    xyz=np.zeros((9,9,3),dtype=float)
    xyz[:,:,2]=40+np.arange(9)[None,:]*3
    return Observation(Path('unused'),{},xyz)


def test_edge_height_spread_can_be_disabled_without_changing_estimator():
    obs=observation()
    with pytest.raises(DualArmError,match='discontinuity'):obs.sample([4,4],8)
    assert np.array_equal(obs.sample([4,4],None),[0,0,52])


def test_missing_depth_and_image_boundary_still_rejected():
    obs=observation();obs.xyz[4,4]=np.nan
    with pytest.raises(DepthUnavailableError):obs.sample([4,4],None)
    obs=observation();obs.xyz[2:4,2:7]=np.nan
    with pytest.raises(DepthUnavailableError):obs.sample([4,4],None)
    with pytest.raises(DualArmError,match='interior'):observation().sample([0,0],None)


def test_oneshot_disables_only_grasp_spread(monkeypatch,tmp_path):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    calls=[]
    monkeypatch.setattr(episode,'prepare_collection_action',lambda *a,**k:calls.append(k))
    episode.prepare_action(None,None,{},tmp_path)
    assert calls[0]['check_grasp_height_spread'] is False
    assert calls[0]['target_sampler'] is episode.table_target_depth.project_target
    assert calls[0]['orientation_resolver'] is episode.one_shot_orientation.resolve
