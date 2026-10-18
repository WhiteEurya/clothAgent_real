"""Choose the nearer half-turn-equivalent parallel-jaw task orientation."""
import ast
import numpy as np
from scipy.spatial.transform import Rotation
from .codex_motion import parse_code


def nearest_grasp_orientation(source, reference_world_rotation, *, prefer_ccw=False, arm='right'):
    calls=parse_code(source,allow_grasp=True)
    if calls[0][0]!='grasp' or set(calls[0][1])!={arm}:return source,None
    reference=Rotation.from_matrix(reference_world_rotation)
    grasp=Rotation.from_euler('xyz',calls[0][1][arm][3:])
    flip=Rotation.from_rotvec([0,0,np.pi])
    original=(reference.inv()*grasp).magnitude()
    alternative=(reference.inv()*grasp*flip).magnitude()
    changed=alternative+1e-8<original
    def twist_degrees(rotation):
        # Tool-local swing/twist decomposition; negative follows the arm7 CCW convention.
        q=rotation.as_quat()
        if np.hypot(q[2],q[3])<1e-8:return None
        return float((np.degrees(2*np.arctan2(q[2],q[3]))+180)%360-180)
    twists=[twist_degrees(reference.inv()*grasp),twist_degrees(reference.inv()*grasp*flip)]
    if prefer_ccw and all(t is not None for t in twists):
        # Already aligned is preferable to an unnecessary half-turn.
        options=[i for i,t in enumerate(twists) if -180-1e-6<=t<=1e-6]
        if options:changed=bool(min(options,key=lambda i:abs(twists[i])))
    report=dict(mode='parallel_jaw_half_turn',applied=changed,
                prefer_ccw=prefer_ccw,view='arm_toward_gripper',
                candidate_local_twist_deg=twists,selected_local_twist_deg=twists[int(changed)],
                original_rotation_deg=float(np.degrees(original)),
                equivalent_rotation_deg=float(np.degrees(alternative)),
                preserves='XYZ and relative orientation throughout grasp/transport')
    if not changed:return source,report
    tree=ast.parse(source)
    for node,(_,args) in zip(tree.body,calls):
        pose=list(args[arm])
        pose[3:]=(Rotation.from_euler('xyz',pose[3:])*flip).as_euler('xyz').tolist()
        next(k for k in node.value.keywords if k.arg==arm).value=ast.parse(repr(pose),mode='eval').body
    return ast.unparse(ast.fix_missing_locations(tree))+'\n',report
