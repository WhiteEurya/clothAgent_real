from types import SimpleNamespace
import pytest
from PIL import Image
from cloth_agent.experiments import graph_minimal_grounding as mg


def fixture(tmp_path):
    path=tmp_path/'reference.png';Image.new('RGB',(100,80),'gray').save(path)
    t=dict(reference_image=str(path),reference_sha256=mg.sha(path),image_size=[100,80],
           seed_grasp_px=[20,30],seed_target_px=[70,50],spec=dict(edge_id='e004'))
    edge=dict(edge_id='e004',replayable=True,policy_outcome='PROGRESS')
    return t,edge


def test_reference_dots_and_original_unchanged(tmp_path):
    t,e=fixture(tmp_path);out=mg.marked_seed_reference(t,e,tmp_path/'output')
    im=Image.open(out)
    assert im.getpixel((20,30))==(255,0,0)
    assert im.getpixel((70,50))==(0,0,255)
    assert mg.sha(t['reference_image'])==t['reference_sha256']


@pytest.mark.parametrize('bad',['checksum','bounds','negative'])
def test_invalid_reference_rejected(tmp_path,bad):
    t,e=fixture(tmp_path)
    if bad=='checksum':t['reference_sha256']='wrong'
    if bad=='bounds':t['seed_grasp_px']=[100,30]
    if bad=='negative':e['policy_outcome']='NO_PROGRESS'
    with pytest.raises(ValueError):mg.marked_seed_reference(t,e,tmp_path/'output')


@pytest.mark.parametrize('edge_id,count',[('e004',2),('e001',1)])
def test_ground_attaches_reference_only_for_e004(tmp_path,edge_id,count):
    t,e=fixture(tmp_path);e['edge_id']=edge_id;t['spec']['edge_id']=edge_id
    g=mg.MinimalGrounder.__new__(mg.MinimalGrounder);g.template=lambda edge:t
    g.expand=lambda value,*args:value
    expected=dict(grasp_px=[25,35],target_px=[65,45],confidence=.9)
    class Model:
        def ask(self,kind,payload,images,validator):
            assert len(images)==count and images[0][0]=='current'
            if count==2:
                assert images[1][0]=='successful_seed_reference'
                assert '不要复制绝对像素位置' in payload['reference_instruction']
            else:assert 'reference_instruction' not in payload
            validator(expected);return expected
    assert g.ground(Model(),e,SimpleNamespace(image=tmp_path/'current.png'),tmp_path/'out')==expected
