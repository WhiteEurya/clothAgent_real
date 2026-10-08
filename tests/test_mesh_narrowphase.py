import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from cloth_agent.dual_arm.geometry import Capsule, DualArmError, check_collision
from cloth_agent.dual_arm.mesh_narrowphase import check_cell_collision, separating_gap
from .test_dual_arm_runtime import scene,ROOT


class HullModel:
    def __init__(self,x):
        self.points=np.array(list(itertools.product([-.5,.5],repeat=3)))+[x,0,0]
        self.config=SimpleNamespace(capsules=[{'name':'link1','radius_mm':10.}])
    def collision_hull(self,q,name):
        return self.points,np.vstack([np.eye(3),-np.eye(3)]),0.


def test_mesh_separation_resolves_coarse_capsules_without_ignoring_margins():
    models={'left':HullModel(0),'right':HullModel(10)}
    config=SimpleNamespace(obstacles=[],limits={'clearance_mm':0})
    centers={k:np.zeros(1) for k in models}
    caps={k:[Capsule('link1',m.points.mean(0),m.points.mean(0),10)] for k,m in models.items()}
    with pytest.raises(DualArmError):check_collision(caps,[],0)
    check_cell_collision(config,models,centers,caps)
    for shapes in caps.values():shapes[0].radius+=5
    with pytest.raises(DualArmError):check_cell_collision(config,models,centers,caps)


def test_mesh_uncertainty_covers_off_axis_rotation():
    models={'left':HullModel(0),'right':HullModel(10)}
    config=SimpleNamespace(obstacles=[],limits={'clearance_mm':0})
    centers={k:np.zeros(1) for k in models}
    caps={k:[Capsule('link1',m.points.mean(0),m.points.mean(0),10)] for k,m in models.items()}
    with pytest.raises(DualArmError):
        check_cell_collision(config,models,centers,caps,
                             half_ranges={k:np.array([30.]) for k in models})


def test_projection_gap_never_reports_intersecting_boxes_separated():
    a=HullModel(0).points;b=HullModel(.5).points
    assert separating_gap(a,np.eye(3),b,np.eye(3))<=0


@pytest.mark.parametrize('gap',[.1,-.1])
def test_directional_proof_does_not_inflate_horizontal_motion_vertically(scene,gap):
    import json
    from cloth_agent.dual_arm.kinematics import ArmModel
    from cloth_agent.dual_arm.stop_sweep import certify_joint_box
    records=json.loads((ROOT/'data/robot/dual_arm_home.json').read_text())['arms']
    q={k:np.asarray(records[i]['joints']) for i,k in enumerate(scene.config.arms)}
    scene.config.arms['right'].world_from_base[0,3]=5000
    scene.config.limits['clearance_mm']=0.
    models={k:ArmModel(a) for k,a in scene.config.arms.items()}
    cap=models['left'].capsules(q['left'])[0]
    z=cap.start[2]+cap.radius+gap
    scene.config.obstacles=[{'name':'ceiling','min_mm':[-10000,-10000,z],
                            'max_mm':[10000,10000,z+1],
                            'excluded_capsules':['right/'+c['name'] for c in scene.config.arms['right'].capsules]}]
    lo={k:v.copy() for k,v in q.items()};hi={k:v.copy() for k,v in q.items()}
    lo['left'][0]-=10;hi['left'][0]+=10
    if gap>0:
        # Full 20-degree range proven in ONE cell despite a 0.1-mm vertical gap.
        assert certify_joint_box(scene.config,models,lo,hi)['nodes']==1
    else:
        with pytest.raises(DualArmError,match='contains collision'):
            certify_joint_box(scene.config,models,lo,hi)
