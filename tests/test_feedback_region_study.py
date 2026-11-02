import json
import pytest
from cloth_agent.cloth_harness.ad_region_study import score
from cloth_agent.cloth_harness.region_review import RegionStore,page
from cloth_agent.cloth_harness.review_app import FIELDS


def test_partial_region_labels_do_not_make_unmarked_points_wrong():
    label={'next_decision':'CONTINUE','regions':[{'kind':'allow','rect':[.1,.1,.2,.2]},{'kind':'avoid','rect':[.6,.6,.2,.2]}]}
    assert score({'decision':'ACTION','grasp_px':[20,20]},label,[100,100])=='ACCEPTABLE'
    assert score({'decision':'ACTION','grasp_px':[70,70]},label,[100,100])=='UNSUITABLE'
    assert score({'decision':'ACTION','grasp_px':[45,45]},label,[100,100])=='UNLABELLED'
    assert score({'decision':'ABSTAIN'},label,[100,100])=='ABSTAIN'
    assert score({'decision':'DONE'},label,[100,100])=='UNSUITABLE'
    assert score({'decision':'DONE'},{'next_decision':'UNCERTAIN'},[100,100])=='UNLABELLED'


def test_region_edit_revokes_both_confirmations(tmp_path):
    (tmp_path/'abcd').mkdir();(tmp_path/'abcd/protocol_ABCD.json').write_text(json.dumps({'selected_cases':['S01']}));(tmp_path/'pairs.json').write_text(json.dumps([{'id':'S01'}]))
    store=RegionStore(tmp_path)
    data={'id':'S01','labels':{k:'UNKNOWN' for k,_,_ in FIELDS},'reviewer':'test','note':'','next_decision':'CONTINUE','regions':[{'kind':'allow','rect':[.1,.1,.2,.2]}],'confirm':False}
    assert not store.save(data)['confirmed']
    data['confirm']=True;assert store.save(data)['confirmed']['S01']
    data['confirm']=False;assert store.save(data)['confirmed']['S01']
    data['regions']=[{'kind':'allow','rect':[.2,.2,.2,.2]}];assert not store.save(data)['confirmed']
    assert not json.loads((tmp_path/'oracle_labels.json').read_text())['S01']['confirmed_by']
    assert not json.loads((tmp_path/'next_action_labels.json').read_text())['S01']['confirmed_by']


def test_overlapping_opposite_regions_rejected(tmp_path):
    (tmp_path/'abcd').mkdir();(tmp_path/'abcd/protocol_ABCD.json').write_text(json.dumps({'selected_cases':['S01']}));(tmp_path/'pairs.json').write_text(json.dumps([{'id':'S01'}]))
    store=RegionStore(tmp_path)
    with pytest.raises(ValueError,match='重叠'):
        store.save({'id':'S01','regions':[{'kind':'allow','rect':[.1,.1,.2,.2]},{'kind':'avoid','rect':[.2,.2,.2,.2]}]})


def test_review_has_canvas_before_initialization():
    html=page('test').decode();assert '<canvas id="regions"' in html
    assert html.index('const originalRender=render')<html.index("api('/data').then")
    assert '__TOKEN__' not in html and '__FIELDS__' not in html
