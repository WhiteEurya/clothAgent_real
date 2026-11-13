from pathlib import Path
import numpy as np
import pytest
from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.self_collision import SelfCollisionChecker
from cloth_agent.dual_arm.geometry import DualArmError

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/dual_arm.dual_wrist_xarm7_20261110.json'
STOPPED=np.array([-2.451457,-72.086513,-36.619280,-5.776159,-38.681527,67.808237,3.281157])

@pytest.fixture(scope='module')
def checkers():
    scene=DualConfig.load_geometry(CONFIG,root=ROOT)
    return {k:SelfCollisionChecker(ArmModel(a),scene.raw['geometry_margin_mm'][k],scene.limits['clearance_mm'])
            for k,a in scene.arms.items()}


def test_real_c22_posture_rejected_without_increasing_existing_margin(checkers):
    c=checkers['right']
    assert c.required==10
    with pytest.raises(DualArmError,match=r'link1/0 vs link6/0'):
        c.check_box(STOPPED,np.zeros(7))


@pytest.mark.parametrize('arm',['left','right'])
def test_home_and_its_tracking_envelope_remain_clear(checkers,arm):
    c=checkers[arm]
    c.check_box(c.model.config.home_joints,np.full(7,.2))


def test_pair_topology_does_not_ignore_nonadjacent_links(checkers):
    pairs={(a['host'],b['host']) for a,b,_ in checkers['right'].pairs}
    assert ('link1','link6') in pairs and ('link2','link4') in pairs
    assert ('link1','link2') not in pairs


def test_swept_box_rejects_an_unsafe_interior(checkers):
    c=checkers['right']
    # Center is the real rejected posture. Large boxes cannot be certified by
    # testing endpoints alone; interval checks must include their interior.
    with pytest.raises(DualArmError,match='self-collision'):
        c.check_box(STOPPED,np.array([0,20,0,0,0,0,0]))
