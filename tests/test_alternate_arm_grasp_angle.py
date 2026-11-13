from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cloth_agent.replay_dual_arm_runtime import PartitionedReplayRuntime
from cloth_agent.dual_arm.geometry import pose_matrix, DualArmError
from cloth_agent.codex_motion import parse_code


def test_rotated_grasp_preserves_world_opening_axis_and_points():
    homes={'left':[0,0,300,180,0,10], 'right':[0,0,300,180,0,170]}
    arms={k:SimpleNamespace(world_from_base=np.eye(4),home_joints=[],tcp_offset=[0,0,0,0,0,v]) for k,v in [('left',15),('right',-20)]}
    models={k:SimpleNamespace(forward=lambda q,k=k:homes[k]) for k in homes}
    runtime=SimpleNamespace(config=SimpleNamespace(arms=arms),models=models,
        _compile_for_arm=lambda source,obs,active_arm:(source,active_arm))
    first=Rotation.from_euler('z',35,degrees=True)*Rotation.from_euler('xyz',homes['left'][3:],degrees=True)
    points=[[.4,.1,.1],[.4,.1,.2],[.5,.1,.2]]
    source='\n'.join(f'motion.{"grasp" if i==0 else "move_tcp"}(right={p+first.as_euler("xyz").tolist()!r})' for i,p in enumerate(points))
    result,arm=PartitionedReplayRuntime._alternate_candidate(runtime,source,None,'left','right')
    assert arm=='right'
    calls=parse_code(result,allow_grasp=True)
    desired=first.apply(Rotation.from_euler('xyz',arms['left'].tcp_offset[3:],degrees=True).inv().apply([0,1,0]))
    for (_,args),point in zip(calls,points):
        assert args['right'][:3]==pytest.approx(point)
        rotation=Rotation.from_euler('xyz',args['right'][3:])
        actual=rotation.apply(Rotation.from_euler('xyz',arms['right'].tcp_offset[3:],degrees=True).inv().apply([0,1,0]))
        assert abs(np.dot(actual[:2]/np.linalg.norm(actual[:2]),desired[:2]/np.linalg.norm(desired[:2])))==pytest.approx(1)
    tilted=Rotation.from_euler('x',20,degrees=True)*first
    bad='\n'.join(f'motion.{name}(right={point+tilted.as_euler("xyz").tolist()!r})' for name,point in [('grasp',points[0]),('move_tcp',points[1])])
    with pytest.raises(DualArmError,match='non-yaw'):
        PartitionedReplayRuntime._alternate_candidate(runtime,bad,None,'left','right')
