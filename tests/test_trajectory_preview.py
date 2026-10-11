from types import SimpleNamespace
import json
import threading
import pytest

from cloth_agent.trajectory_preview import TrajectoryPreview
from .test_dual_arm_runtime import scene
from .test_codex_native_runtime import runtime_for


@pytest.mark.parametrize('moved',[False,True])
def test_preview_is_before_dispatch_and_arrival_is_rechecked(scene,tmp_path,moved):
    runtime=runtime_for(scene)
    shown=threading.Event();approve=threading.Event();results=[]
    class Preview:
        def show(self,record,path):
            assert all(not c.commands for c in scene.connections.values())
            self.record=record
        def await_execution(self,cancel):
            shown.set()
            assert approve.wait(5)
    runtime.preview=Preview()
    obs=scene.obs;obs.meta['synthetic']=False
    thread=threading.Thread(target=lambda:results.append(runtime.execute('motion.gripper(arm="left",action="open")',obs,tmp_path)))
    try:
        thread.start();assert shown.wait(5)
        assert results==[] and all(not c.commands for c in scene.connections.values())
        assert json.loads((tmp_path/'trajectory_preview.json').read_text())['status']=='VERIFIED'
        if moved:scene.connections['left'].joints[0]+=2
        approve.set();thread.join(5);assert not thread.is_alive()
        assert results[0]['status']==('PLANNING_FAILED' if moved else 'EXECUTED')
        if moved:
            assert all(not c.commands for c in scene.connections.values())
            assert runtime.preview.record['status']=='REJECTED'
    finally:
        approve.set();thread.join(5);runtime.close()


def test_failed_plan_saves_exact_witness_and_does_not_wait(scene,tmp_path,monkeypatch):
    from cloth_agent.dual_arm.stop_sweep import UnsafeBoxError
    runtime=runtime_for(scene);q={k:a.home_joints.tolist() for k,a in scene.config.arms.items()}
    report=dict(status='UNSAFE',witness=dict(objects=['left/camera','right/camera'],joints_deg=q,gap_mm=-.03),
                segment=dict(start_deg=q,end_deg=q))
    def fail(source,initial):
        runtime.planning_context={'command_index':1,'route':'tcp_direct'}
        error=UnsafeBoxError(report);runtime._preview_failure(error);raise error
    monkeypatch.setattr(runtime,'plan',fail)
    received=[]
    runtime.preview=SimpleNamespace(show=lambda r,p:received.append(r),await_execution=lambda *_:pytest.fail('unsafe plan offered for execution'))
    scene.obs.meta['synthetic']=False
    try:
        result=runtime.execute('motion.move_tcp(left=[.25,-.12,.18,0,0,0])',scene.obs,tmp_path)
        assert result['status']=='PLANNING_FAILED'
        saved=json.loads((tmp_path/'trajectory_preview.json').read_text())
        assert saved['failures'][0]['report']==report
        assert saved['targets'][0]['world_pose_mm_deg'][:3]==[250,-120,180]
        assert received[0]['status']=='REJECTED'
        assert all(not c.commands for c in scene.connections.values())
    finally:runtime.close()


def test_stale_or_rejected_preview_button_cannot_approve():
    preview=object.__new__(TrajectoryPreview)
    preview.lock=threading.RLock();preview.ready=threading.Event()
    preview.execute=SimpleNamespace(disabled=False)
    preview.record={'status':'VERIFIED'}
    preview._approve(SimpleNamespace(target=object()))
    assert not preview.ready.is_set()
    preview.record['status']='REJECTED'
    preview._approve(SimpleNamespace(target=preview.execute))
    assert not preview.ready.is_set()
    preview.record['status']='VERIFIED'
    preview._approve(SimpleNamespace(target=preview.execute))
    assert preview.ready.is_set() and preview.execute.disabled


def test_home_transition_also_waits_for_preview(scene,tmp_path,monkeypatch):
    runtime=runtime_for(scene);seen=[]
    target=scene.config.arms['left'].home_joints.copy();target[0]+=1
    monkeypatch.setattr(runtime,'perception_target',lambda:('left',target))
    def show(record,path):
        assert record['status']=='VERIFIED'
        assert all(not c.commands for c in scene.connections.values())
        seen.append('show')
    def wait(cancel):
        assert seen==['show']
        assert all(not c.commands for c in scene.connections.values())
        seen.append('wait')
    runtime.preview=SimpleNamespace(show=show,await_execution=wait)
    try:
        runtime.move_to_perception(tmp_path/'prepare')
        assert seen==['show','wait']
        assert scene.connections['left'].commands
    finally:runtime.close()


def test_preview_wait_is_cancellable():
    preview=object.__new__(TrajectoryPreview)
    preview.wait=True;preview.ready=threading.Event()
    cancel=threading.Event();cancel.set()
    with pytest.raises(KeyboardInterrupt,match='preview cancelled'):preview.await_execution(cancel)


def test_automatic_preview_does_not_wait_for_approval():
    preview=object.__new__(TrajectoryPreview)
    preview.wait=False
    preview.ready=SimpleNamespace(wait=lambda *_:pytest.fail('automatic mode must not wait'))
    preview.await_execution(threading.Event())


@pytest.mark.parametrize('kind',['exact','approximate','missing','joint'])
def test_target_model_is_diagnostic_and_preserves_other_arm(scene,monkeypatch,kind):
    import numpy as np
    from cloth_agent.trajectory_preview import target_configuration
    from cloth_agent.dual_arm.geometry import DualArmError,pose_matrix,matrix_pose
    q={k:a.home_joints.copy() for k,a in scene.config.arms.items()}
    target_q=q['right'].copy();target_q[0]+=1
    world=scene.config.arms['right'].world_from_base@pose_matrix(scene.models['right'].forward(target_q))
    target=dict(arm='right',command_index=1,world_pose_mm_deg=matrix_pose(world).tolist())
    record=dict(initial=q,status='REJECTED')
    if kind=='joint':
        record['source']=f'motion.move_joints(right={np.radians(target_q).tolist()})'
        monkeypatch.setattr(scene.models['right'],'inverse',lambda *a:pytest.fail('joint target must use saved branch'))
    elif kind in ('approximate','missing'):
        def reject(*args):
            error=DualArmError('no exact IK')
            if kind=='approximate':error.ik_candidate=dict(joints_deg=q['right'].tolist())
            raise error
        monkeypatch.setattr(scene.models['right'],'inverse',reject)
    result=target_configuration(record,target,scene.models,scene.config)
    assert record['status']=='REJECTED'
    assert all(not c.commands for c in scene.connections.values())
    if kind=='missing':
        assert result['status']=='UNAVAILABLE' and result['joints'] is None
    else:
        assert np.array_equal(result['joints']['left'],q['left'])
        assert result['status']==('APPROXIMATE' if kind=='approximate' else 'IK_OK')
        assert result['distance_mm']==pytest.approx(1 if kind=='approximate' else 0,abs=1e-9)


def test_point_cloud_uses_saved_world_coordinates_and_matching_colors(tmp_path):
    import numpy as np
    from PIL import Image
    from cloth_agent.trajectory_preview import observation_files,load_preview_cloud
    before=tmp_path/'before';before.mkdir()
    xyz=np.array([[[700.,-80.,25.],[np.nan,0.,0.]],[[600.,-40.,30.],[500.,10.,40.]]])
    rgb=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],dtype=np.uint8)
    np.save(before/'world_xyz_mm.npy',xyz);Image.fromarray(rgb).save(before/'rgb.png')
    points,colors=load_preview_cloud(observation_files(tmp_path),max_points=2)
    assert np.allclose(points,[[.7,-.08,.025],[.5,.01,.04]])
    assert colors.tolist()==[[255,0,0],[255,255,0]]
    assert observation_files(tmp_path/'unrelated') is None
