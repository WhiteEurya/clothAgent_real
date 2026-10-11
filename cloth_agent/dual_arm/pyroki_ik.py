"""PyRoki IK for the arm-only URDF chain, with the calibrated SDK TCP.

Public inputs/outputs match ArmModel (base-frame mm/deg). PyRoki operates in
metres/radians. Collision geometry and final FK acceptance stay in ArmModel.
"""
from copy import deepcopy

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from .geometry import DualArmError, pose_matrix

# Match the existing double-precision kinematic model and sub-mm acceptance.
jax.config.update('jax_enable_x64', True)


@jdc.jit
def _solve(robot, link_index, target, initial, reference, max_iterations: jdc.Static[int] = 100):
    var=robot.joint_var_cls(0)
    costs=[
        pk.costs.pose_cost_analytic_jac(
            robot,var,jaxlie.SE3(target),link_index,pos_weight=50.,ori_weight=10.),
        pk.costs.limit_constraint(robot,var),
        pk.costs.rest_cost(var,rest_pose=reference,weight=.001),
    ]
    solution=jaxls.LeastSquaresProblem(costs, [var]).analyze().solve(
        initial_vals=jaxls.VarValues.make((var.with_value(initial),)),
        verbose=False,linear_solver='dense_cholesky',
        trust_region=jaxls.TrustRegionConfig(lambda_initial=1.),
        termination=jaxls.TerminationConfig(max_iterations=max_iterations),
    )
    return solution[var]


class PyrokiIK:
    def __init__(self,model):
        # Extract precisely link_base -> physical flange. Gripper/mimic joints
        # cannot become extra free IK variables. Never modify the live FK URDF.
        parents={j.child:j for j in model.urdf.robot.joints}
        joints=[];links=[];name=model.flange
        while name!='link_base':
            links.append(deepcopy(model.urdf.link_map[name]))
            joint=parents[name];joints.append(deepcopy(joint));name=joint.parent
        links.append(deepcopy(model.urdf.link_map['link_base']))
        joints.reverse()
        links.reverse()
        tcp_name='clothagent_ik_tcp'
        links.append(yourdfpy.Link(name=tcp_name))
        offset=pose_matrix(model.config.tcp_offset);offset[:3,3]/=1000.
        joints.append(yourdfpy.Joint(name='clothagent_ik_tcp_joint',type='fixed',
            parent=model.flange,child=tcp_name,origin=offset))
        urdf=yourdfpy.URDF(yourdfpy.Robot(name='clothagent_ik',links=links,joints=joints),
            load_meshes=False,load_collision_meshes=False)
        self.max_iterations=getattr(model,'ik_max_iterations',100)
        self.robot=pk.Robot.from_urdf(urdf)
        names=self.robot.joints.actuated_names
        if set(names)!=set(model.names):
            raise DualArmError(f'PyRoki arm joint mismatch: {names}')
        self.to_pyroki=np.array([model.names.index(n) for n in names])
        self.from_pyroki=np.array([names.index(n) for n in model.names])
        self.link_index=jnp.asarray(self.robot.links.names.index(tcp_name),dtype=jnp.int32)

    def solve(self,pose_mm_deg,seed_deg,reference_deg):
        target=pose_matrix(pose_mm_deg);target[:3,3]/=1000.
        target=jaxlie.SE3.from_matrix(jnp.asarray(target)).wxyz_xyz
        seed=jnp.asarray(np.radians(seed_deg)[self.to_pyroki])
        reference=jnp.asarray(np.radians(reference_deg)[self.to_pyroki])
        solved=_solve(self.robot,self.link_index,target,seed,reference,self.max_iterations)
        return np.degrees(np.asarray(solved)[self.from_pyroki])

    def forward(self,joints_deg):
        """Independent TCP FK, for cross-checking the adapter in offline tests."""
        cfg=jnp.asarray(np.radians(joints_deg)[self.to_pyroki])
        frame=np.array(jaxlie.SE3(self.robot.forward_kinematics(cfg)[self.link_index]).as_matrix())
        frame[:3,3]*=1000.
        return frame
