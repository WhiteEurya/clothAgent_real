import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from cloth_agent.experiments.action_overlay_review import project,projection,draw_overlay,model_row,run
from cloth_agent.experiments.discovery_data import save,file_hash,digest
from cloth_agent.experiments.decision_state_discovery import Requests


def fixture(tmp_path):
    raw=tmp_path/'perception';raw.mkdir();before=tmp_path/'before';before.mkdir()
    im=Image.new('RGB',(100,80),'gray');im.save(raw/'camera_0_A.png')
    im.transpose(Image.Transpose.ROTATE_270).save(before/'camera_A_rgb_upright.png')
    k=[[100,0,50],[0,100,40],[0,0,1]];x=np.eye(4).tolist()
    pts=[[0,0,1000],[100,0,1000],[0,100,1000]]
    save(raw/'camera_A_coordinate_guide.json',dict(coordinate_frame='robot_base_mm',samples=[dict(base_xyz_mm=p,pixel_xy=q) for p,q in zip(pts,[[50,40],[60,40],[50,50]])]))
    save(raw/'result.json',dict(views=[dict(label='A',image='camera_0_A.png',intrinsics=k,X_base_camera=x)]))
    save(before/'camera_A_upright_mapping.json',dict(rotation='clockwise90',coordinate_guide=str(raw/'camera_A_coordinate_guide.json'),raw_size_xy=[100,80],upright_size_xy=[80,100]))
    p=before/'camera_A_rgb_upright.png'
    return dict(sample_id='s',before_rgb=str(p),image_hashes={'before':file_hash(p)},action=dict(grasp=pts[0],target=pts[1],normalized=dict(delta=[100,0,0]),raw=[]))


def test_recorded_projection_rotation_and_no_double_offset(tmp_path):
    s=fixture(tmp_path);v=projection(s)
    assert v['Pa_pixel_xy']==[39,50] and v['Pb_pixel_xy']==[39,60]
    assert v['displacement_pixel_xy']==[0,10]
    assert v['calibration_check']['max_reprojection_error_px']==0
    x=np.eye(4);x[1,3]=-.02
    assert np.allclose(project([[0,-20,1000]],[[100,0,50],[0,100,40],[0,0,1]],x),[[50,40]])
    with pytest.raises(ValueError,match='behind'):project([[0,0,-100]],np.eye(3),np.eye(4))


def test_projection_refuses_mismatched_calibration_and_image(tmp_path):
    s=fixture(tmp_path);f=tmp_path/'perception/result.json';v=json.loads(f.read_text());v['views'][0]['intrinsics'][0][2]+=5;save(f,v)
    with pytest.raises(ValueError,match='reprojection'):projection(s)
    Image.new('RGB',(100,80),'red').save(tmp_path/'perception/camera_0_A.png')
    with pytest.raises(ValueError,match='does not match'):projection(s)


def test_overlay_leaves_original_and_model_payload_whitelisted(tmp_path):
    s=fixture(tmp_path);p=projection(s);original=file_hash(s['before_rgb']);draw_overlay(s['before_rgb'],tmp_path/'overlay.png','s',p)
    assert file_hash(s['before_rgb'])==original
    row=dict(s,projection=p,image_only_assignment={},outcome='SECRET',after_rgb='SECRET',evaluation='SECRET',run_id='SECRET')
    text=json.dumps(model_row(row));assert 'SECRET' not in text and 'sources' not in text and 'calibration' not in text
    assert (tmp_path/'overlay.png').is_file()


class Fake:
    def __init__(self):self.calls=[]
    def invoke(self,prompt,images,schema):
        p=json.loads(prompt.split('Payload:\n')[1].split('\nSchema:\n')[0]);ids=p['sample_ids'];self.calls.append((p,images))
        assert 'Do not regenerate' in prompt
        assert all(Path(x).exists() for x in images)
        v=dict(covered_sample_ids=ids,sample_assessments=[dict(sample_id=s,grasp_explanation='fixture',target_explanation='fixture',displacement_explanation='fixture',geometry_explanatory_power='UNKNOWN',action_family='fixture',unexplained_action_variation=[],confidence=.5) for s in ids],common_action_families=[],supporting_evidence=[],appearance_similar_action_different=[],appearance_different_action_equivalent=[],proposed_additional_factors=[],needs_additional_factor='UNRESOLVED',conclusion='fixture',limitations=[])
        return dict(text=json.dumps(v),elapsed_s=.01,usage={})


def test_frozen_review_multimodal_coverage_and_cache(tmp_path):
    s=fixture(tmp_path);p=projection(s);draw_overlay(s['before_rgb'],tmp_path/'overlays/s.png','s',p)
    tax=dict(factors=[dict(id='target_geometry')],classes=[])
    save(tmp_path/'frozen_taxonomy.json',tax);save(tmp_path/'frozen_taxonomy.sha256.json',dict(sha256=file_hash(tmp_path/'frozen_taxonomy.json')))
    row=dict(s,projection=p,image_only_assignment={});plan=dict(samples=[row],batch_size=12,frozen_taxonomy=tax,frozen_taxonomy_hash=digest(tax),task_goal={})
    b=Fake();r=Requests(tmp_path,'gpt-6-astra',b);out=run(plan,tmp_path,r)
    assert out['covered_sample_ids']==['s'] and len(b.calls)==2
    assert len(b.calls[0][1])==2
    run(plan,tmp_path,r);assert len(b.calls)==2
    assert json.loads((tmp_path/'frozen_taxonomy.json').read_text())==tax
    assert not (tmp_path/'stage2/taxonomy_v1.json').exists()


def test_report_distinguishes_new_final_pairs_from_verified_pairs(tmp_path):
    from cloth_agent.experiments.action_overlay_review import report
    pair=lambda a,b:dict(samples=[a,b],evidence='fixture',uncertainty='fixture')
    v=dict(appearance_similar_action_different=[pair('a','b'),pair('b','c')],appearance_different_action_equivalent=[])
    save(tmp_path/'stage2/action_review.json',v)
    save(tmp_path/'stage2/pair_check_000.json',dict(appearance_similar_action_different=[pair('b','a')],appearance_different_action_equivalent=[]))
    p=dict(samples=[],source_manifest_hash='fixture',frozen_taxonomy_hash='fixture',limitations=[])
    r=report(p,tmp_path)
    audit=r['counterexample_verification']['appearance_similar_action_different']
    assert [x['present_in_full_resolution_pair_review'] for x in audit]==[True,False]
