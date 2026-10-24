from pathlib import Path
import numpy as np
from PIL import Image

from cloth_agent.experiments.dino_corpus_search import inventory, make_action_refs


def test_inventory_includes_after_pending_and_failed_and_deduplicates(tmp_path):
    for name in ['s/traj_0000/observation','s/traj_0000/after','s/pending/iter_1/observation']:
        folder=tmp_path/name;folder.mkdir(parents=True)
        Image.new('RGB',(32,24),'red').save(folder/'rgb.png')
    entries,summary=inventory(tmp_path)
    assert summary['file_count']==3 and len(entries)==1
    assert set(a['role'] for a in entries[0]['aliases'])=={'action_before','action_after','pending_or_other'}


def test_only_explicit_pre_image_binds_to_recorded_point_pair():
    entries={h:dict(image=h+'.png',features=h+'.npz',features_sha256='test',geometry={'original_wh':[32,24]}) for h in ['before','after']}
    points=[dict(name='start',pixel_xy=[5,6]),dict(name='target',pixel_xy=[20,10])]
    archive=dict(steps=[dict(step_id='s1',session='session',source_folder='/session/traj_0000',
        before_observation='o1',after_observation='o2',commanded_action=dict(selected_points=points))],
        observations={'o1':{'rgb':{'sha256':'before','path':'/before.png'}},'o2':{'rgb':{'sha256':'after','path':'/after.png'}}})
    refs,excluded=make_action_refs(archive,entries,[])
    assert len(refs)==1 and refs[0]['source_image']['sha256']=='before'
    assert refs[0]['edge']['seed_action']['exact_command']['selected_points']==points
    assert not excluded
    archive['steps'][0]['commanded_action']['selected_points']=[]
    refs,excluded=make_action_refs(archive,entries,[])
    assert not refs and excluded[0]['reason']=='no_unique_Pa_Pb_pair'
