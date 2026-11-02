import numpy as np
import pytest
from cloth_agent.ggo.observer import Settings,warp
from cloth_agent.ggo.assessment import boundary_profiles,summarize_asymmetry,assessment,write_outputs


def profiles(current=None,coverage=None,angle=0):
    ref=np.zeros((360,360),bool);ref[90:230,90:230]=True
    if current is None:current=ref.copy();current[:,215:]=False
    frame={'center':np.array([320.,180.]),'ex':np.array([1.,0.]),'ey':np.array([0.,1.])}
    pts={'HL':np.array([180,460,0]),'HR':np.array([460,460,0])};cfg=Settings()
    a=np.deg2rad(angle);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    center=np.array([360.,360.]);t=center-R@center
    cur=warp(current,np.zeros(2),cfg,R,t)
    obs=np.ones_like(cur) if coverage is None else warp(coverage,np.zeros(2),cfg,R,t)
    return boundary_profiles(ref,cur,obs,np.zeros(2),frame,pts,cfg,R,t)


@pytest.mark.parametrize('angle',[0,45,90,-90])
def test_signed_profile_uses_garment_frame_not_image_y(angle):
    d=profiles(angle=angle)
    valid=[s for s in d['samples'] if s['right']['status']=='MEASURED']
    assert len(valid)>20
    assert all(abs(s['right']['inward_mm']-30)<=4 for s in valid)
    assert all(abs(s['left']['inward_mm'])<=4 for s in valid)
    assert d['summary']['right']['longest_measured_inward_span_mm']>=200


def test_depth_gap_is_unknown_and_breaks_continuous_contraction():
    coverage=np.ones((360,360),bool);coverage[135:160,210:220]=False
    d=profiles(coverage=coverage)
    gap=[s for s in d['samples'] if 100<s['v_mm']<130]
    assert gap and all(s['right']['inward_mm'] is None for s in gap)
    assert d['summary']['right']['longest_measured_inward_span_mm']<200


def test_multiple_intersections_not_matched_to_nearest_edge():
    current=np.zeros((360,360),bool);current[90:230,90:215]=True;current[130:160,200:205]=False
    d=profiles(current=current)
    row=next(s for s in d['samples'] if s['v_mm']==100)
    assert row['right']['status']=='UNMEASURABLE'


def trial(left,right):
    return {'regions':{'LEFT_LOWER_BODY':{'missing_ratio':left,'missing_area_mm2':left*10000 if left is not None else None},
                       'RIGHT_LOWER_BODY':{'missing_ratio':right,'missing_area_mm2':right*10000 if right is not None else None}}}


def test_stability_requires_sign_and_margin_in_all_trials():
    assert summarize_asymmetry([trial(.1,.4),trial(.2,.35)],.05)['relation']=='RIGHT_GREATER_THAN_LEFT'
    flip=summarize_asymmetry([trial(.1,.4),trial(.3,.1)],.05)
    assert flip['status']=='UNCERTAIN'
    assert summarize_asymmetry([trial(.1,.4),trial(None,.3)],.05)['status']=='UNKNOWN'
    assert summarize_asymmetry([trial(.1,.12),trial(.1,.09)],.05)['status']=='NO_CLEAR_ASYMMETRY'


def test_unknown_registration_never_produces_definite_assessment(tmp_path):
    r={'registration':{'status':'UNKNOWN'},'deformation':{'status':'UNKNOWN'},'corners':{'right_hem':{'status':'UNVERIFIED','point':None}}}
    value=write_outputs(tmp_path,r)
    assert value['lower_body']['right']['projected_deficit']=='UNKNOWN'
    assert value['asymmetry']['status']=='UNKNOWN'
    assert value['registration']['status']=='UNRELIABLE'
    assert (tmp_path/'geometry_assessment.json').exists()
    assert (tmp_path/'body_cross_sections.csv').read_text().count('\n')==1


def test_area_level_independent_of_contour_label_and_curve_is_exported():
    low={'min':0.,'max':.001,'valid_trials':27,'total_trials':27}
    regions={key:{'missing_ratio':0.,'missing_area_mm2':0.,'reliability':'POSE_SENSITIVE','state':'UNCERTAIN'} for key in ('LEFT_LOWER_BODY','RIGHT_LOWER_BODY')}
    report={'registration':{'status':'OK'},'settings':{'contraction_ratio':.12,'tolerance_mm':6},
            'deformation':{'regions':regions,'registration_uncertainty':{'regional_ranges':{key:{'missing_ratio':low} for key in regions},'asymmetry':{'status':'NO_CLEAR_ASYMMETRY','relation':'WITHIN_MARGIN'}}},
            'body_cross_sections':profiles()}
    a=assessment(report)
    assert a['lower_body']['left']['projected_deficit']=='LOW'
    assert a['interpretation']=='NO_SIGNIFICANT_LOWER_BODY_PROJECTED_DEFICIT'
    assert len(a['section_curve']['samples'])>20
