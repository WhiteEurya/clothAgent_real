import json
from types import SimpleNamespace
import numpy as np
from cloth_agent.experiments.unfold_planning_skills import workspace_prompt


def test_projected_bounds_match_base_bounds_and_failed_target(tmp_path):
    world_left=np.eye(4);world_left[:3,:3]=np.diag([-1,-1,1]);world_left[0,3]=1058
    transforms={'right':np.eye(4),'left':world_left}
    workspaces={
        'right':dict(min_mm=[None,None,-20],max_mm=[577.506409,127.638359,None]),
        'left':dict(min_mm=[None,None,8],max_mm=[None,None,None],halfspaces_mm=[[1,-.02,.017,577.47]])}
    runtime=SimpleNamespace(config_path='test.json',config=SimpleNamespace(
        raw={'arms':{k:dict(workspace=workspaces[k],world_from_base_mm=t.tolist()) for k,t in transforms.items()}},
        arms={k:SimpleNamespace(world_from_base=t) for k,t in transforms.items()}),
        models={k:SimpleNamespace(forward=lambda q:np.array([200,0,450,0,0,0])) for k in transforms})
    before=SimpleNamespace(meta={'joints_deg':{'left':[], 'right':[]}})
    (tmp_path/'view.json').write_text(json.dumps(dict(mm_per_pixel=2.641265061598558,
        center_world_xy_mm=[560.7696533203125,-303.46942138671875])))
    from PIL import Image
    Image.new('RGB',(640,640)).save(tmp_path/'rgb.png')
    text=workspace_prompt(runtime,before,SimpleNamespace(directory=tmp_path,plan_coordinates='topdown',image=tmp_path/'rgb.png'),tmp_path)
    report=json.loads((tmp_path/'planning_workspace.json').read_text())
    world=np.asarray(report['pixel_to_world'])
    point=np.array([330,205,58,1.])
    right=np.asarray(report['arms']['right']['pixel_world_z_halfspaces'])
    assert np.any(right[:,:3]@point[:3]>right[:,3])
    for name,t in transforms.items():
        planes=np.asarray(report['arms'][name]['pixel_world_z_halfspaces'])
        base=np.linalg.inv(t)@world@point
        ws=workspaces[name]
        expected=all((lo is None or val>=lo) and (hi is None or val<=hi)
                     for val,lo,hi in zip(base[:3],ws['min_mm'],ws['max_mm']))
        expected &= all(np.dot(p[:3],base[:3])<=p[3] for p in ws.get('halfspaces_mm',[]))
        assert bool(np.all(planes[:,:3]@point[:3]<=planes[:,3]))==expected
    assert 'right:' in text and 'left:' in text and '放置时不换臂' in text
    assert '326.336644' in text
