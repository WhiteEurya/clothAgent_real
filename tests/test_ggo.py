import numpy as np
import pytest
from cloth_agent.ggo.observer import Settings, register, rigid_fit, transform, compare, warp, fit_table, table_cloud, sample_anchor


def case():
    mask=np.zeros((260,260),bool);mask[40:210,70:190]=True;mask[40:90,25:235]=True
    regions={
        'LEFT_LOWER_BODY':np.zeros_like(mask),'RIGHT_LOWER_BODY':np.zeros_like(mask),
        'LEFT_SLEEVE':np.zeros_like(mask),'RIGHT_SLEEVE':np.zeros_like(mask)}
    regions['LEFT_LOWER_BODY'][90:210,70:130]=True
    regions['RIGHT_LOWER_BODY'][90:210,130:190]=True
    regions['LEFT_SLEEVE'][40:90,25:70]=True
    regions['RIGHT_SLEEVE'][40:90,190:235]=True
    paths={'LEFT_LOWER_BODY':np.c_[np.full(120,70),np.arange(90,210)],
           'RIGHT_LOWER_BODY':np.c_[np.full(120,189),np.arange(90,210)],
           'LEFT_SLEEVE':np.c_[np.full(50,25),np.arange(40,90)],
           'RIGHT_SLEEVE':np.c_[np.full(50,234),np.arange(40,90)]}
    return np.pad(mask,40),{k:np.pad(v,40) for k,v in regions.items()},{k:v*2+81 for k,v in paths.items()}


def test_rigid_fit_rotation_translation_no_scale_and_outlier():
    a=np.array([[0,0],[-100,80],[100,80],[-80,40],[80,40.]])
    ang=.7;R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
    b=transform(a,R,[120,-50]);b[-1]+=[300,100]
    result=register(a,b)
    assert result['status']=='OK' and result['inlier_count']==4 and result['scale']==1
    np.testing.assert_allclose(result['rotation'],R,atol=1e-8)
    np.testing.assert_allclose(result['translation_mm'],[120,-50],atol=1e-8)
    assert np.linalg.det(result['rotation'])==pytest.approx(1)
    scaled=register(a[:3],a[:3]*.6)
    assert scaled['status']!='OK'


def test_low_support_unknown_and_missing_anchors():
    a=np.array([[0,0],[2,2],[4,0.]])
    assert register(a,a)['status']!='OK'
    assert register(a[:2],a[:2])['status']=='UNKNOWN'


def test_refinement_cannot_destroy_valid_three_point_consensus():
    a=np.array([[0.,0.],[-120.,80.],[120.,80.]])
    b=a.copy();b[0,1]+=17.
    result=register(a,b)
    assert result['status']=='OK'
    assert result['refinement_retained_initial']
    assert max(result['residuals_mm'])<=12.
    b[0,1]+=80.
    rejected=register(a,b)
    assert rejected['status']=='UNKNOWN'
    assert rejected['reason']=='fewer than three geometrically consistent anchors'


def test_boundary_tolerance_measures_cloth_not_background_and_keeps_missing_unknown():
    y,x=np.indices((30,30));xyz=np.stack([x,y,np.zeros_like(x)],axis=-1).astype(float)
    mask=np.zeros((30,30),bool);mask[:,15:]=True;xyz[mask,2]=10.
    audit={}
    point=sample_anchor(xyz,mask,[12,15],tolerance_px=3,audit=audit)
    assert point[2]==10. and audit['status']=='BOUNDARY_CORRECTED'
    assert audit['offset_px']==3. and audit['sampled_px']==[15,15]
    assert sample_anchor(xyz,mask,[11,15],tolerance_px=3) is None
    assert sample_anchor(xyz,mask,None,tolerance_px=3) is None
    xyz[15,16]=np.nan
    assert sample_anchor(xyz,mask,[16,15],tolerance_px=3) is None
    assert sample_anchor(xyz,mask,[12,15]) is None  # strict default for references


@pytest.mark.parametrize('angle',[0,45,90,180,-90])
def test_flat_rigid_rotation_has_no_structural_deficit(angle):
    mask,regions,paths=case();cfg=Settings();rad=np.deg2rad(angle)
    R=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    center=np.array([340.,340.]);t=center-R@center
    current=warp(mask,np.zeros(2),cfg,R,t)
    report,_=compare(regions,paths,current,np.ones_like(mask),np.zeros(2),cfg,R,t)
    assert report['primary_region'] is None
    assert all(v['missing_ratio']==0 for v in report['regions'].values())
    assert all(v['contour_deviation_mm']<3 for v in report['regions'].values())


@pytest.mark.parametrize('region',['LEFT_LOWER_BODY','RIGHT_LOWER_BODY','LEFT_SLEEVE','RIGHT_SLEEVE'])
def test_local_missing_region_detected_without_any_current_hem_point(region):
    mask,regions,paths=case();current=mask.copy();current[regions[region]]=False
    report,_=compare(regions,paths,current,np.ones_like(mask),np.zeros(2),Settings(),np.eye(2),np.zeros(2))
    assert report['primary_region']==region
    assert report['regions'][region]['state']=='SUSPECTED_CONTRACTION'
    assert report['regions'][region]['largest_missing_component_mm2']>600


def test_missing_depth_does_not_become_missing_material():
    mask,regions,paths=case();current=mask.copy();observed=np.ones_like(mask)
    current[regions['RIGHT_LOWER_BODY']]=False;observed[regions['RIGHT_LOWER_BODY']]=False
    report,_=compare(regions,paths,current,observed,np.zeros(2),Settings(),np.eye(2),np.zeros(2))
    assert report['regions']['RIGHT_LOWER_BODY']['missing_ratio'] is None
    assert report['regions']['RIGHT_LOWER_BODY']['state']=='UNKNOWN'
    assert report['primary_region'] is None


def test_plane_height_and_missing_anchor():
    y,x=np.indices((120,160));z=.1*x+.05*y+20;xyz=np.stack([x,y,z],axis=-1).astype(float)
    mask=np.zeros((120,160),bool);mask[40:80,65:95]=True;xyz[mask,2]+=10
    table=fit_table(xyz,mask);cloud=table_cloud(xyz,table)
    assert np.median(cloud[~mask,2])==pytest.approx(0,abs=.01)
    assert np.median(cloud[mask,2])==pytest.approx(10/np.sqrt(1+.1**2+.05**2),abs=.01)
    xyz[50,80]=np.nan
    assert sample_anchor(xyz,mask,[80,50]) is None
    assert sample_anchor(xyz,mask,None) is None


def test_end_to_end_observer_flat_and_right_contraction_without_current_corner():
    from cloth_agent.ggo.observer import observe
    y,x=np.indices((400,500));xyz=np.stack([x,y,np.zeros_like(x)+20],axis=-1).astype(float)
    mask=np.zeros(x.shape,bool);mask[70:140,100:400]=True;mask[110:320,160:340]=True
    rgb=np.full((*mask.shape,3),35,np.uint8);rgb[mask]=[180,120,200];xyz[mask,2]+=2
    reference={'N':[250,80],'AL':[162,138],'AR':[337,138],'SL':[103,100],'SR':[396,100],'HL':[163,316],'HR':[336,316]}
    ann=dict(reference=reference,current={k:reference[k] for k in ['N','AL','AR']},provenance='synthetic test fixture')
    flat,_=observe(rgb,xyz,rgb,xyz,ann)
    assert flat['registration']['status']=='OK'
    assert flat['deformation']['primary_region'] is None
    assert flat['model_calls']==0
    current=rgb.copy();current[180:320,270:340]=35
    cx=xyz.copy();cx[180:320,270:340,2]=20
    folded,art=observe(rgb,xyz,current,cx,ann)
    assert folded['registration']['status']=='OK'
    assert folded['deformation']['primary_region']=='RIGHT_LOWER_BODY'
    assert folded['corners']['right_hem']['status']=='UNVERIFIED'
    assert folded['corners']['right_hem']['point'] is None
    assert art['missing'].sum()>0
    import json
    json.dumps(folded,allow_nan=False)
    # Current upper-body anchor occlusion must suppress regional inference.
    ann['current']['AR']=None
    unknown,_=observe(rgb,xyz,current,cx,ann)
    assert unknown['registration']['status']=='UNKNOWN'
    assert unknown['deformation']['status']=='UNKNOWN'


def test_printed_letter_is_not_a_cloth_hole_but_depth_remains_unknown():
    from cloth_agent.ggo.observer import silhouette
    rgb=np.full((120,160,3),35,np.uint8);rgb[20:100,40:120]=[180,120,200]
    rgb[40:60,60:80]=240
    mask=silhouette(rgb)
    assert mask[50,70] and not mask[10,70]
    xyz=np.ones((120,160,3));xyz[48:53,68:73]=np.nan
    assert sample_anchor(xyz,mask,[70,50]) is None
