import copy

from PIL import Image
import pytest

from cloth_agent.experiments import local_dense_proposal_probe as p


def manifest_and_labels():
    m=dict(queries=[dict(id='q01',size=[1280,720]),dict(id='q02',size=[1280,720])])
    a=dict(annotation_origin='HUMAN_REQUIRED',annotator='',queries={q['id']:dict(label='UNREVIEWED',regions=[],reviewed_all_regions=False) for q in m['queries']})
    return m,a


def valid_labels():
    m,a=manifest_and_labels();a['annotator']='test-human'
    a['queries']['q01']=dict(label='POSITIVE',regions=[dict(type='rectangle',polygon_px=[[10,20],[80,20],[80,90],[10,90]])],reviewed_all_regions=True)
    a['queries']['q02']=dict(label='NOT_APPLICABLE',regions=[],reviewed_all_regions=True)
    return m,a


def test_no_proxy_or_missing_labels_accepted():
    m,a=manifest_and_labels();assert p.annotation_errors(m,a)
    m,a=valid_labels();assert p.annotation_errors(m,a)==[]
    for change in [dict(annotation_origin='ASSISTANT_PREANNOTATED'),dict(annotator='')]:
        assert p.annotation_errors(m,dict(a,**change))
    b=copy.deepcopy(a);b['queries']['q01']['reviewed_all_regions']=False
    assert p.annotation_errors(m,b)
    b=copy.deepcopy(a);b['queries']['q02']['regions']=a['queries']['q01']['regions']
    assert p.annotation_errors(m,b)


def test_freeze_is_manual_gate_and_cannot_overwrite(tmp_path):
    m,a=manifest_and_labels();m.update(files={},protocol_sha256='')
    p.save(tmp_path/'protocol.json',{'feature_computation_performed':False});m['protocol_sha256']=p.sha(tmp_path/'protocol.json')
    p.save(tmp_path/'manifest.json',m);a['manifest_sha256']=p.sha(tmp_path/'manifest.json')
    p.save(tmp_path/'annotations.draft.json',a)
    with pytest.raises(ValueError):p.freeze(tmp_path)
    assert not (tmp_path/'annotations.frozen.json').exists()
    _,a=valid_labels();a['manifest_sha256']=p.sha(tmp_path/'manifest.json');p.save(tmp_path/'annotations.draft.json',a)
    target=p.freeze(tmp_path);assert target.exists()
    assert p.read(tmp_path/'annotation_freeze.json')['feature_computations']==0
    with pytest.raises(ValueError,match='已冻结'):p.freeze(tmp_path)


def test_pixel_duplicates_ignore_file_encoding(tmp_path):
    im=Image.new('RGB',(80,90),'purple');a=tmp_path/'a.png';b=tmp_path/'b.png'
    im.save(a,compress_level=0);im.save(b,compress_level=9)
    assert p.sha(a)!=p.sha(b) and p.pixel_hash(a)==p.pixel_hash(b)


def test_mapping_and_canvas_coordinates():
    m=p.resize_mapping((1280,720))
    assert m['resized_size']==[896,504] and m['grid_wh']==[64,36]
    assert m['padding_ltrb']==[0,0,0,0]
    assert p.canvas_to_original(250,120,.5,(10,20),(1280,720))==[480,200]


def test_selection_has_session_coverage_without_labels():
    groups={s:[dict(session=s,pixel_sha256=f'{s}_{i}') for i in range(10)] for s in ['demo']+[f's{i}' for i in range(10)]}
    selected=p.select_queries(groups,'demo')
    assert len(selected)==16 and len({r['pixel_sha256'] for r in selected})==16
    assert sum(r['session']=='demo' for r in selected)==4
    assert len({r['session'] for r in selected})==11
