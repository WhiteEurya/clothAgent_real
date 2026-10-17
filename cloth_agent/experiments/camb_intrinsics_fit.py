"""Offline multi-view RGB intrinsic fit; candidates only, never deploy."""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from pupil_apriltags import Detector
from cloth_agent.experiments.apriltag_geometry_audit import board_config,DEFAULT_BOARD,A4_SCALE,fit_pose,project_rgb

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1))))

def fit(views,k,size,distorted=False):
    flags=cv2.CALIB_USE_INTRINSIC_GUESS
    if not distorted:flags|=cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3
    else:flags|=cv2.CALIB_FIX_K3
    out=cv2.calibrateCameraExtended([v['object'].astype(np.float32) for v in views],
        [v['pixels'].astype(np.float32) for v in views],tuple(size),k.copy(),np.zeros(5),flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,100,1e-8))
    rms,K,d,rv,tv,std_i,std_e,perview=out
    return dict(train_rms_px=float(rms),K=K.tolist(),dist=d.ravel().tolist(),
        parameter_stddev=std_i.ravel().tolist(),per_view_rms_px=perview.ravel().tolist())

def heldout_error(views,K,d):
    errors=[];per=[]
    for view in views:
        # Fit pose on half the tags, evaluate disjoint tag corners.
        mask=np.repeat(np.arange(len(view['ids']))%2==0,4)
        ok,rv,tv=cv2.solvePnP(view['object'][mask],view['pixels'][mask],K,d,flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:raise RuntimeError('Held-out pose fit failed')
        predicted=cv2.projectPoints(view['object'][~mask],rv,tv,K,d)[0].reshape(-1,2)
        e=np.linalg.norm(predicted-view['pixels'][~mask],axis=1)
        errors.extend(e);per.append(dict(image=view['image'],rms_px=float(np.sqrt(np.mean(e**2)))))
    return dict(rms_px=float(np.sqrt(np.mean(np.square(errors)))),median_px=float(np.median(errors)),views=per)

def analyze(folder):
    folder=Path(folder);out=folder/'intrinsics_fit';out.mkdir(exist_ok=False)
    meta=json.loads((folder/'camera.json').read_text());frames=json.loads((folder/'frames.json').read_text())
    k=np.asarray(meta['K'],float);size=meta['image_size'];board=board_config(DEFAULT_BOARD,A4_SCALE)
    detector=Detector(families=board['family'],quad_decimate=1,nthreads=2)
    accepted=[];audit=[];previous=None
    for frame in frames:
        rgb=cv2.imread(str(folder/frame['image']))
        tags=detector.detect(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY))
        row=dict(**frame,detected=len(tags))
        tags=[t for t in tags if t.tag_id in board['corners'] and t.hamming==0 and t.decision_margin>=20]
        if len({t.tag_id for t in tags})!=len(tags):row['rejected']='DUPLICATE_BOARD_IDS';previous=None
        elif len(tags)<8:row['rejected']='FEWER_THAN_8_TAGS';previous=None
        else:
            tags=sorted(tags,key=lambda t:t.tag_id)
            ids=[t.tag_id for t in tags];obj=np.concatenate([board['corners'][i] for i in ids]);pix=np.concatenate([t.corners for t in tags])
            pose,amb=fit_pose(obj,pix,k,np.zeros(5));normal=pose[:3,2]
            if normal[2]<0:normal=-normal
            error=np.linalg.norm(project_rgb(pose,obj,k,np.zeros(5))-pix,axis=1)
            rms=float(np.sqrt(np.mean(error**2)))
            center=pix.mean(axis=0);sizepx=float(np.mean([np.linalg.norm(t.corners[1]-t.corners[0]) for t in tags]))
            byid={t.tag_id:t.corners for t in tags}
            common=set(byid)&set(previous or {})
            movement=float(np.sqrt(np.mean(np.square(np.concatenate([byid[i]-previous[i] for i in common]))))) if len(common)>=8 else 999.
            previous=byid
            row.update(rms_px=rms,movement_px=movement,normal=normal.tolist(),center=center.tolist(),tag_size_px=sizepx)
            duplicate=any(angle(normal,v['normal'])<6 and np.linalg.norm(center-v['center'])<35
                and abs(sizepx/v['sizepx']-1)<.12 for v in accepted)
            if rms>2:row['rejected']='REPROJECTION_OVER_2PX'
            elif sizepx<16:row['rejected']='TAG_TOO_SMALL'
            elif movement>2:row['rejected']='NOT_STATIONARY_PAIR'
            elif duplicate:row['rejected']='REDUNDANT_POSE'
            else:
                row['selected']=True
                accepted.append(dict(image=frame['image'],object=obj,pixels=pix,ids=ids,normal=normal,center=center,sizepx=sizepx))
        audit.append(row)
    (out/'frame_audit.json').write_text(json.dumps(audit,indent=2))
    (out/'selected_views.json').write_text(json.dumps([{key:(v.tolist() if isinstance(v,np.ndarray) else v)
        for key,v in a.items()} for a in accepted],indent=2))
    normals=np.array([v['normal'] for v in accepted]);centers=np.array([v['center'] for v in accepted])
    coverage=dict(selected_views=len(accepted))
    reasons=[]
    if len(accepted)<12:reasons.append('少于 12 个停稳且不重复的板姿态')
    if len(accepted):
        tilts=np.degrees(np.arctan2(normals[:,:2],normals[:,2:]))
        coverage.update(tilt_x_span_deg=float(np.ptp(tilts[:,0])),tilt_y_span_deg=float(np.ptp(tilts[:,1])),
            center_x_span_fraction=float(np.ptp(centers[:,0])/size[0]),
            center_y_span_fraction=float(np.ptp(centers[:,1])/size[1]),
            max_normal_separation_deg=max(angle(a,b) for a in normals for b in normals))
        if min(coverage['tilt_x_span_deg'],coverage['tilt_y_span_deg'])<15:reasons.append('至少一个倾斜方向跨度不足 15 度')
        if coverage['max_normal_separation_deg']<20:reasons.append('板法向变化不足 20 度')
        if coverage['center_x_span_fraction']<.2 or coverage['center_y_span_fraction']<.15:reasons.append('板中心在图像中的位置覆盖不足')
    else:reasons.append('无可用姿态')
    # Montage selected frames, with one shared factory reference.
    thumb_w=320;thumb_h=200;montage=Image.new('RGB',(thumb_w*4,thumb_h*max(1,(len(accepted)+3)//4)),(30,30,30))
    for index,v in enumerate(accepted):
        image=Image.open(folder/v['image']).convert('RGB');image.thumbnail((320,180))
        x=index%4*thumb_w;y=index//4*thumb_h;montage.paste(image,(x,y))
        ImageDraw.Draw(montage).text((x+4,y+180),f'{index}: {Path(v["image"]).name}',fill='white')
    montage.save(out/'selected_montage.jpg')
    report=dict(status='INSUFFICIENT_DIVERSITY' if reasons else 'CANDIDATE_ONLY',coverage=coverage,reasons=reasons,
        factory=meta,calibration_modified=False,depth_used=False,
        selection_thresholds=dict(min_tags=8,min_views=12,max_motion_px=2,min_axis_tilt_span_deg=15,min_normal_span_deg=20))
    if not reasons:
        train=[v for i,v in enumerate(accepted) if i%4!=3];test=[v for i,v in enumerate(accepted) if i%4==3]
        report['train_images']=[v['image'] for v in train];report['test_images']=[v['image'] for v in test]
        models={'factory':dict(K=k.tolist(),dist=meta['dist'])}
        for name,distorted in [('pinhole',False),('brown4',True)]:
            models[name]=fit(train,k,size,distorted)
            subsets=[fit([v for i,v in enumerate(train) if i%3!=fold],k,size,distorted)['K'] for fold in range(3)]
            array=np.array(subsets);models[name]['subset_K']=subsets
            models[name]['subset_focal_span_percent']=((np.ptp(array[:,[0,1],[0,1]],axis=0)/np.diag(np.asarray(models[name]['K']))[:2])*100).tolist()
        for model in models.values():model['heldout']=heldout_error(test,np.asarray(model['K']),np.asarray(model['dist']))
        report['models']=models
    (out/'report.json').write_text(json.dumps(report,indent=2))
    lines=['# Camera B RGB 内参离线诊断','',f'状态：{report["status"]}','',
        f'采集 {len(frames)} 帧；选中 {len(accepted)} 个停稳且不同的姿态。','',
        json.dumps(coverage,ensure_ascii=False,indent=2),'',*reasons]
    if 'models' in report:
        lines+=['','| Model | fx | fy | cx | cy | held-out RMS px |','|---|---:|---:|---:|---:|---:|']
        for name,m in report['models'].items():
            K=np.asarray(m['K']);lines.append(f'| {name} | {K[0,0]:.3f} | {K[1,1]:.3f} | {K[0,2]:.3f} | {K[1,2]:.3f} | {m["heldout"]["rms_px"]:.3f} |')
    lines+=['','没有修改或部署任何内参。只使用 RGB，不使用 depth、world 外参或高度补偿。',
        'train/test 按去重后的姿态分开；每个测试姿态用一半 tag 拟合外姿态，用另一半角点计分。',
        '这是候选拟合，需结合子集稳定性与 held-out 误差判断；参数形式为 OpenCV Brown，不能直接替换 RealSense inverse-Brown 元数据。',
        '统一打印尺度影响板距离，不影响这类内参拟合；非等比打印、弯曲和角点误差仍会影响内参。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def combine_sessions(sessions):
    sessions=[Path(s).resolve() for s in sessions]
    first=json.loads((sessions[0]/'camera.json').read_text())
    out=sessions[0].parent/('combined_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    frames=[]
    for session in sessions:
        meta=json.loads((session/'camera.json').read_text())
        if meta['serial']!=first['serial'] or meta['image_size']!=first['image_size'] or not np.allclose(meta['K'],first['K']):
            raise ValueError('Capture camera/intrinsics/resolution changed')
        for frame in json.loads((session/'frames.json').read_text()):
            frames.append(dict(frame,image=str(session/frame['image']),source_session=str(session)))
    out.mkdir()
    (out/'camera.json').write_text(json.dumps(first,indent=2))
    (out/'frames.json').write_text(json.dumps(frames,indent=2))
    (out/'source_sessions.json').write_text(json.dumps([str(s) for s in sessions],indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('session',nargs='+');a=p.parse_args()
    analyze(combine_sessions(a.session) if len(a.session)>1 else a.session[0])
