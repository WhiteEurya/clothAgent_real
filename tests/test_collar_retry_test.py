import numpy as np
import pytest
from cloth_agent.experiments.collar_retry_test import geometry,laydown_points

def test_base_retreat_table_advance_and_height():
 g=geometry([400,0,40],[0,0],[600,0],table_z=20)
 assert g['lifted']==[400,0,290]
 assert g['retreated']==[340,0,290]
 assert g['target']==[460,0,25]
 assert g['contact_source']=='UNVERIFIED_OPEN_LOOP'
 pts=np.array(laydown_points([340,0,150],g['target']))
 assert np.all(np.diff(pts[:,2])<0)
 assert np.all(np.diff(pts[:,0])>0)
 assert np.max(np.linalg.norm(np.diff(np.vstack([[340,0,150],pts]),axis=0),axis=1))<=20+1e-9
 assert np.allclose(pts[-1],g['target'])

@pytest.mark.parametrize('point,base,center', [([400,0,19],[0,0],[600,0]),([0,0,40],[0,0],[600,0]),([400,0,float('nan')],[0,0],[600,0])])
def test_bad_geometry_blocks(point,base,center):
 with pytest.raises(ValueError):geometry(point,base,center,table_z=20)

def test_laydown_cannot_ascend():
 with pytest.raises(ValueError):laydown_points([0,0,20],[180,0,25])


def test_thin_collar_uses_five_mm_offset_above_hard_table():
 g=geometry([577.5,16.5,25.6],[0,0],[500,0],table_z=23)
 assert g["grasp_surface"][2]==25.6
 assert g["grasp_offset_mm"]==-5
 assert g["grasp_xyz_mm"][2]==pytest.approx(20.6)


def test_forward_reverses_retreat_even_past_nominal_center():
 g=geometry([577,13,26],[0,0],[500,0],table_z=23)
 retreat=np.array(g['retreated'])[:2]-np.array(g['lifted'])[:2]
 forward=np.array(g['target'])[:2]-np.array(g['retreated'])[:2]
 assert np.dot(retreat,forward)<0
 assert np.allclose(forward,-2*retreat)


def test_automatic_tail_descends_then_lays_down_and_releases():
 from cloth_agent.experiments.collar_retry_test import automatic_laydown_tail
 from cloth_agent.codex_motion import parse_code
 g=geometry([577,13,26],[0,0],[500,0],table_z=23)
 source,contact=automatic_laydown_tail(g,[180,0,0])
 ops=parse_code(source)
 poses=[np.array(v['left'][:3])*1000 for k,v in ops if k=='move_tcp']
 assert poses[0][2]==pytest.approx(123)
 assert poses[-2][2]==pytest.approx(28)
 assert poses[-1][2]==pytest.approx(58)
 assert np.allclose(poses[-2],g['target'])
 assert ops[-2][0]=='gripper' and ops[-2][1]['action']=='open'
 assert contact['contact_verified'] is False
 assert all(poses[i+1][2]<=poses[i][2] for i in range(len(poses)-2))

def test_live_runner_has_no_operator_input():
 import ast,inspect
 from cloth_agent.experiments.collar_retry_test import run_live
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='input' for n in ast.walk(ast.parse(inspect.getsource(run_live))))

def test_diagnostic_identifies_interpolated_lift_target(tmp_path):
 from types import SimpleNamespace
 from cloth_agent.experiments.collar_retry_test import failure_diagnostics
 from cloth_agent.dual_arm.geometry import DualArmError
 e=DualArmError('IK rejected')
 e.ik_failure=dict(command_index=5,phase='tcp_direct_ik',target_world_m_rad=[.6,.1,.28,0,0,0],seed_joints_deg=[1]*6,required_position_error_mm=.5,required_orientation_error_deg=.2)
 rt=SimpleNamespace(planning_context=dict(command_index=5,cartesian_fraction=.4),plan_stats=dict(dispatched_segments=0))
 d=failure_diagnostics(rt,e,[dict(command_index=5,phase='lift')])
 assert d['phase']=='lift'
 assert d['failed_target_world_mm_deg'][:3]==pytest.approx([600,100,280])
 assert d['planning_context']['cartesian_fraction']==.4
 assert d['dispatched_segments']==0
 assert d['thresholds_modified'] is False

def test_command_stage_mapping_includes_release():
 from cloth_agent.experiments.collar_retry_test import command_stages,automatic_laydown_tail
 from cloth_agent.gamepad_collect import tcp_command
 g=geometry([577,13,26],[0,0],[500,0],table_z=23)
 move=tcp_command(np.array([577,13,26,180,0,0]))
 prefix='motion.gripper(arm="left", action="open")\n'+'motion.move_joints(left=[0,0,0,0,0,0])\n'+move+'motion.gripper(arm="left", action="close")\n'+move+move
 tail,_=automatic_laydown_tail(g,[180,0,0]);stages=command_stages(prefix+tail)
 assert stages[4]['phase']=='lift'
 assert stages[5]['phase']=='retreat'
 assert stages[6]['phase']=='vertical_descent'
 assert stages[-2]['phase']=='release' and stages[-2]['command']=='gripper'
 assert stages[-1]['phase']=='post_release_retreat'

def test_home_height_is_absolute_world_z_and_preserves_xy():
 g=geometry([625,87,43],[0,0],[500,0],table_z=23,lift_world_z=480)
 assert g['lifted']==[625,87,480]
 assert g['retreated'][2]==480
 assert g['lift_mm']==437
 assert g['lift_mode']=='HOME_WORLD_Z'
 assert g['grasp_xyz_mm']==[625,87,38]
 with pytest.raises(ValueError):geometry([625,87,43],[0,0],[500,0],lift_world_z=40)

def test_home_stop_only_retracts_and_rises_without_release(monkeypatch):
 import cloth_agent.experiments.collar_retry_test as retry
 from cloth_agent.codex_motion import parse_code
 monkeypatch.setattr(retry,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\nmotion.move_joints(left=[0,0,0,0,0,0])\n')
 home=np.array([400,0,370,180,0,0])
 source,stages,g=retry.home_stop_program(None,[638,7,26],home,23)
 calls=parse_code(source)
 assert len(calls)==5
 assert calls[-2]==('gripper',dict(arm='left',action='close'))
 assert calls[-1][0]=='move_tcp'
 assert np.allclose(calls[-1][1]['left'][:3],[.4,0,.37])
 assert stages[-1]['phase']=='lift_retract_to_home'
 assert not g['release'] and not g['return_to_perception']

def test_vertical_drop_keeps_pick_xy_and_opens_only_at_top(monkeypatch):
 import cloth_agent.experiments.collar_retry_test as retry
 from cloth_agent.codex_motion import parse_code
 monkeypatch.setattr(retry,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\nmotion.move_joints(left=[0,0,0,0,0,0])\n')
 source,stages,g=retry.vertical_drop_program(None,[638,7,26],[494,-3,370,180,0,0],23)
 calls=parse_code(source)
 assert len(calls)==6
 assert calls[-1]==('gripper',dict(arm='left',action='open'))
 assert calls[-3]==('gripper',dict(arm='left',action='close'))
 assert np.allclose(calls[-2][1]['left'][:3],[.638,.007,.370])
 assert np.allclose(calls[2][1]['left'][:3],[.638,.007,.021])
 assert stages[-2]['phase']=='vertical_lift_to_home_height'
 assert g['horizontal_displacement_mm']==0 and g['release']
 assert not g['return_to_perception']

def test_shake_only_z_before_release_with_no_extra_gripper_commands(monkeypatch):
 import cloth_agent.experiments.collar_retry_test as retry
 from cloth_agent.codex_motion import parse_code
 monkeypatch.setattr(retry,'approach_program',lambda *a:'motion.gripper(arm="left", action="open")\nmotion.move_joints(left=[0,0,0,0,0,0])\n')
 source,stages,g=retry.vertical_drop_program(None,[500,7,26],[494,-3,370,180,0,0],23,shake_cycles=2)
 calls=parse_code(source)
 assert len(calls)==10
 assert [s['phase'] for s in stages[5:9]]==['shake_down','shake_up','shake_down','shake_up']
 assert np.allclose([calls[i][1]['left'][:3] for i in range(5,9)],[[.5,.007,.36],[.5,.007,.37],[.5,.007,.36],[.5,.007,.37]])
 assert calls[-1]==('gripper',dict(arm='left',action='open'))
 assert g['shake_cycles']==2
 with pytest.raises(ValueError):retry.vertical_drop_program(None,[500,7,26],[494,-3,370,180,0,0],23,shake_cycles=4)


def test_live_runner_does_not_reference_main_local_args():
 import ast,inspect
 from cloth_agent.experiments.collar_retry_test import run_live
 assert not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id=='a' for n in ast.walk(ast.parse(inspect.getsource(run_live))))

def test_shake_speed_only_applies_to_shake_and_preserves_cap():
 from types import SimpleNamespace
 from cloth_agent.experiments.collar_retry_test import shake_speed_profile
 seen=[]
 rt=SimpleNamespace(config=SimpleNamespace(limits={'cartesian_speed_mm_s':5}),planning_context={'command_index':5})
 rt._tcp_segments=lambda *a,**kw:seen.append(rt.config.limits['cartesian_speed_mm_s'])
 original=rt._tcp_segments
 with shake_speed_profile(rt,[dict(command_index=6,phase='shake_down')],30,15):
  rt._tcp_segments()
  rt.planning_context['command_index']=6;rt._tcp_segments()
  assert rt.config.limits['cartesian_speed_mm_s']==5
 assert seen==[5,15] and rt._tcp_segments is original

def test_independent_shake_is_scoped_and_restored_on_error():
 from types import SimpleNamespace
 import pytest
 from cloth_agent.experiments.collar_retry_test import shake_speed_profile
 limits={'cartesian_speed_mm_s':5,'cartesian_accel_mm_s2':30,'joint_speed_deg_s':5}
 rt=SimpleNamespace(config=SimpleNamespace(limits=limits.copy()),planning_context={'command_index':5})
 seen=[]
 def segment(**kwargs):
  assert kwargs == ({"apply_joint_speed_cap":False} if rt.planning_context["command_index"]==6 else {})
  seen.append(rt.config.limits.copy())
  if rt.planning_context['command_index']==6:raise RuntimeError('planning failure')
 rt._tcp_segments=segment
 with shake_speed_profile(rt,[dict(command_index=6,phase='shake_down')],150,15,independent=True,acceleration=500):
  rt._tcp_segments()
  rt.planning_context['command_index']=6
  with pytest.raises(RuntimeError):rt._tcp_segments()
  assert rt.config.limits==limits
 assert seen==[limits,dict(limits,cartesian_speed_mm_s=150,cartesian_accel_mm_s2=500)]
 assert rt._tcp_segments is segment


def test_short_stroke_profile_rejects_long_strokes_and_excess_speed():
 import pytest
 from cloth_agent.experiments.collar_retry_test import short_shake_profile
 profile=short_shake_profile(30,150,3)
 assert profile['validation']['source']=='user_instruction'
 assert profile['validation']['stop_excursion_deg'] is None
 for values in ((60,150,3),(30,151,3),(30,150,4)):
  with pytest.raises(ValueError):short_shake_profile(*values)
