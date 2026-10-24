"""Local RGB-D evidence for one-shot planning; no model, camera or robot calls.

V1 is specialized to the existing purple garment on a dark table. Contours are
visible silhouette candidates, NOT anatomical hem paths or hidden cloth lengths.
"""
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION='PURPLE_VISIBLE_GEOMETRY_V1'
REFERENCE=Path(__file__).resolve().parents[1]/'reports/trajectory_memory_full_episode_v1_collection_aligned/round_006/after'
_REFERENCE_CACHE={}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def garment_mask(rgb):
    a=np.asarray(rgb,dtype=float);r,g,b=a.transpose(2,0,1)
    # Explicit color prior, not a general-purpose clothing segmenter.
    mask=((r>g*1.10)&(b>g*1.10)&(b>r*.85)&(g>45)&(r>100)&(b>110)).astype('uint8')
    # Exclude table rails/people at the sides for this fixed-camera setup.
    w=mask.shape[1];mask[:,:int(.18*w)]=0;mask[:,int(.84*w):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n<=1:return np.zeros(mask.shape,dtype='uint8')
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    if stats[idx,cv2.CC_STAT_AREA]<500:return np.zeros(mask.shape,dtype='uint8')
    return (labels==idx).astype('uint8')


def measure_path(pixels,xyz,mask):
    coords=[];valid=[]
    h,w=mask.shape
    for x,y in pixels:
        x,y=int(x),int(y)
        if not (2<=x<w-2 and 2<=y<h-2):
            coords.append([np.nan]*3);valid.append(False);continue
        patch=xyz[y-2:y+3,x-2:x+3]
        ok=np.isfinite(patch).all(2)&(mask[y-2:y+3,x-2:x+3]>0)
        enough=bool(ok.sum()>=18 and np.isfinite(xyz[y,x]).all())
        coords.append(np.median(patch[ok],axis=0) if enough else [np.nan]*3);valid.append(enough)
    coords=np.array(coords);lengths=np.linalg.norm(np.diff(coords,axis=0),axis=1)
    # Reject gaps or layer jumps, never bridge them into a claimed material path.
    segments=np.isfinite(lengths)&(lengths<=12.)
    full=bool(len(segments)>0 and segments.all())
    chord=float(np.linalg.norm(coords[-1]-coords[0])) if valid[0] and valid[-1] else None
    length=float(lengths.sum()) if full else None
    return dict(endpoints_px=[list(map(int,pixels[0])),list(map(int,pixels[-1]))],
        sampled_points=len(pixels),valid_points=sum(valid),valid_segment_fraction=round(float(segments.mean()),3),
        chord_mm=round(chord,1) if chord is not None else None,
        visible_path_mm=round(length,1) if length is not None else None,
        path_over_chord=round(length/chord,3) if full and chord and chord>1 else None,
        status='MEASURED_VISIBLE_CONTOUR' if full else 'INCOMPLETE_DEPTH_OR_LAYER_JUMP')


def analyze(rgb,xyz):
    if xyz.shape!=(*rgb.shape[:2],3):raise ValueError('RGB/XYZ shape mismatch')
    mask=garment_mask(rgb);ys,xs=np.where(mask)
    if len(xs)<500:return dict(status='UNAVAILABLE',reason='purple garment segmentation failed'),mask,[]
    valid=mask.astype(bool)&np.isfinite(xyz).all(2)
    cloud=xyz[valid]
    info=dict(status='PARTIAL',segmentation='largest purple component, 3px opening/5px closing; fixed image x ROI 18%-84%',
        bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
        touches_image_border=bool(xs.min()<=int(.18*mask.shape[1])+2 or ys.min()<=2 or xs.max()>=int(.84*mask.shape[1])-3 or ys.max()>=mask.shape[0]-3),
        depth_valid_fraction=round(float(valid.sum()/mask.sum()),3))
    if len(cloud)<500:
        info.update(reason='insufficient measured garment depth');return info,mask,[]
    # Fixed subsampling bounds CPU work; world XY spans, invariant to in-plane rotation.
    cloud=cloud[::max(1,len(cloud)//10000)]
    xy=cloud[:,:2];center=np.median(xy,axis=0)
    _,axes=np.linalg.eigh(np.cov((xy-center).T));projected=(xy-center)@axes
    spans=np.diff(np.percentile(projected,[2,98],axis=0),axis=0)[0]
    info.update(status='AVAILABLE_WITH_LIMITATIONS',world_xy_pca_spans_mm=sorted(np.round(spans,1).tolist(),reverse=True),
        world_z_p95_minus_p05_mm=round(float(np.diff(np.percentile(cloud[:,2],[5,95]))[0]),1))
    # Erode inward to keep depth patches off the table; does not trace true hem seams.
    inner=cv2.erode(mask,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(inner,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return info,mask,[]
    contour=max(contours,key=cv2.contourArea)[:,0,:]
    if len(contour)<40:return info,mask,[]
    # Four equal contour-index arcs; IDs are per image, never correspondences.
    arcs=[];paths=[]
    for i in range(4):
        start=int(i*len(contour)/4);end=int((i+1)*len(contour)/4)
        indices=list(range(start,end,5))+[end%len(contour)]
        path=contour[np.array(indices)%len(contour)]
        m=measure_path(path,xyz,mask);m['id']=f'C{i+1}'
        arcs.append(m);paths.append(path)
    info['contour_candidates']=arcs
    return info,mask,paths


def reference(goal):
    goal_sha=sha(goal)
    if goal_sha in _REFERENCE_CACHE:return _REFERENCE_CACHE[goal_sha]
    if not (REFERENCE/'rgb.png').exists() or sha(REFERENCE/'rgb.png')!=goal_sha:
        return dict(status='UNAVAILABLE',reason='no verified RGB-D reference for this goal')
    from .dual_arm.observation import Observation
    obs=Observation.load(REFERENCE)
    info,_,_=analyze(np.asarray(Image.open(obs.image).convert('RGB')),obs.xyz)
    info.update(rgb_sha256=goal_sha,source=str(REFERENCE),xyz_sha256=sha(REFERENCE/'world_xyz_mm.npy'))
    _REFERENCE_CACHE[goal_sha]=info
    return info


def build(before,goal,folder):
    """Persist audit/overlay; fail open to RGB-only planning with explicit unavailable status."""
    started=time.perf_counter();folder=Path(folder)
    report=dict(version=VERSION,role='advisory_only',source_rgb_sha256=sha(before.image),
        limitations=['Contour candidates are NOT identified sleeves/hem; verify against RGB.',
            'Hidden/out-of-frame cloth length is not measured; incomplete paths are null.',
            'PCA spans are global silhouette cues, NOT hem width or completion percentage.',
            'Arcs span anatomical corners; path/chord ratio is NOT a fold amount. Arc IDs have no cross-frame correspondence.',
            'Depth variation includes tilt, drape and noise; it is not a wrinkle severity score.',
            'No hard DONE threshold. Geometry cannot prove acquisition or action feasibility.'])
    try:
        rgb=np.asarray(Image.open(before.image).convert('RGB'))
        info,mask,paths=analyze(rgb,np.asarray(before.xyz))
        report['current']=info;report['reference']=reference(goal)
        report['xyz_provenance']=dict(frame=before.meta.get('frame_id'),calibration_id=before.meta.get('calibration_id'),
            source_sha256=before.meta.get('sha256',{}).get('world_xyz_mm.npy'))
        ref=report['reference']
        if info.get('world_xy_pca_spans_mm') and ref.get('world_xy_pca_spans_mm') and not (info.get('touches_image_border') or ref.get('touches_image_border')):
            report['global_span_ratios_to_reference']=[round(a/b,3) if b>0 else None for a,b in zip(info['world_xy_pca_spans_mm'],ref['world_xy_pca_spans_mm'])]
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        colors=['#00ff80','#00ddff','#ffbb00','#ff70ff']
        for path,metric,color in zip(paths,info.get('contour_candidates',[]),colors):
            draw.line([tuple(map(int,pt)) for pt in path],fill=color,width=3)
            x,y=map(int,path[len(path)//2]);label=f"{metric['id']} path={metric['visible_path_mm']}mm chord={metric['chord_mm']}mm"
            x=max(0,min(x,im.width-360));y=max(0,min(y,im.height-20))
            draw.rectangle((x,y,x+360,y+18),fill='black');draw.text((x+2,y+2),label,fill=color)
        im.save(folder/'geometry_overlay.png');Image.fromarray(mask*255).save(folder/'geometry_mask.png')
    except (ValueError,OSError,AttributeError,KeyError,RuntimeError) as exc:
        report.update(status='UNAVAILABLE',error=f'{type(exc).__name__}: {exc}')
    report['wall_s']=round(time.perf_counter()-started,4)
    (folder/'geometry_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return report


def prompt_report(report):
    # Keep geometry payload compact; per-point measurements remain offline artifacts.
    value={k:v for k,v in report.items() if k in ['version','status','error','current','global_span_ratios_to_reference','limitations']}
    # Arbitrary contour arcs remain audit artifacts, not anatomical evidence.
    if 'current' in value:
        value['current']={k:v for k,v in value['current'].items() if k!='contour_candidates'}
    ref=report.get('reference',{})
    value['reference']={k:v for k,v in ref.items() if k in ['status','reason','world_xy_pca_spans_mm','world_z_p95_minus_p05_mm','touches_image_border']}
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))

# Named anatomical chords: never infer these identities from contour arc indices.
LANDMARKS = ('left_hem','left_armpit','left_cuff_lower','left_cuff_upper',
             'right_cuff_upper','right_cuff_lower','right_armpit','right_hem')
STRUCTURE_PROMPT = '''只识别衣物结构点，不规划动作、不判断DONE。第一张当前RGB，第二张展平参考。
分别标出八个点：左下摆角、左腋下、左袖口下端、左袖口上端、右袖口上端、右袖口下端、右腋下、右下摆角。
左右按衣服领口朝上时的图像左右定义，随衣服旋转保持部位对应，不按当前屏幕左右重新命名；袖口上下端分别指朝肩部/朝腋下的一端。
每张图用自己的原始像素坐标。只有清楚可见且部位身份可信时为VISIBLE并给px；隐藏为HIDDEN；证据不足为UNCERTAIN；后二者px必须null。
折痕形成的尖角不自动等于真实下摆角。不要通过展平图、对称性或预期长度补全隐藏点。每点用一句短语说明可见边缘关系依据。
'''
_POINT_SCHEMA = dict(type='object',additionalProperties=False,required=['visibility','px','evidence'],properties={
    'visibility':dict(type='string',enum=['VISIBLE','HIDDEN','UNCERTAIN']),
    'px':dict(anyOf=[dict(type='array',items=dict(type='integer'),minItems=2,maxItems=2),dict(type='null')]),
    'evidence':dict(type='string')})
_FRAME_SCHEMA = dict(type='object',additionalProperties=False,required=list(LANDMARKS),
                     properties={k:_POINT_SCHEMA for k in LANDMARKS})
STRUCTURE_SCHEMA = dict(type='object',additionalProperties=False,required=['current','reference'],
                        properties={k:_FRAME_SCHEMA for k in ('current','reference')})


def measure_structure(rgb, xyz, landmarks):
    """Measured straight chords only; unknown identities/depth stay unknown."""
    mask=garment_mask(rgb);h,w=mask.shape;points={}
    for name in LANDMARKS:
        item=dict(landmarks[name]);p=item['px'];world=None
        status='IDENTITY_UNCERTAIN'
        if item['visibility']=='VISIBLE' and p is not None:
            x,y=p
            if 0<=x<w and 0<=y<h and mask[y,x] and np.isfinite(xyz[y,x]).all():
                patch=xyz[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]
                m=mask[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]>0
                valid=m & np.isfinite(patch).all(2)
                if valid.sum()>=3:
                    world=np.median(patch[valid],axis=0).tolist();status='MEASURED'
                else:status='DEPTH_UNAVAILABLE'
            else:status='OUTSIDE_CLOTH_OR_DEPTH_UNAVAILABLE'
        item.update(world_mm=world,status=status);points[name]=item
    segments=[]
    for a,b in zip(LANDMARKS,LANDMARKS[1:]+LANDMARKS[:1]):
        pa,pb=points[a]['world_mm'],points[b]['world_mm']
        delta=np.asarray(pa)-np.asarray(pb) if pa is not None and pb is not None else None
        segments.append(dict(endpoints=[a,b],xy_mm=round(float(np.linalg.norm(delta[:2])),1) if delta is not None else None,
                             xyz_mm=round(float(np.linalg.norm(delta)),1) if delta is not None else None))
    return dict(points=points,segments=segments)


def build_structure(before, goal, landmarks, folder):
    from .dual_arm.observation import Observation
    # Require the exact RGB-D reference, never pair a different goal with this depth.
    if sha(goal)!=sha(REFERENCE/'rgb.png'):raise ValueError('unverified structural reference')
    ref=Observation.load(REFERENCE)
    images={'current':np.asarray(Image.open(before.image).convert('RGB')),
            'reference':np.asarray(Image.open(ref.image).convert('RGB'))}
    result=dict(version='ANATOMICAL_CHORDS_V1',role='identity_cross_check_not_proof',
                source_rgb_sha256=sha(before.image),reference_rgb_sha256=sha(goal),
                limitation='Short chords can mean folding OR misidentification. No hidden endpoints, material lengths, DONE threshold or calibrated identity probability.')
    for key,obs in [('current',before),('reference',ref)]:
        result[key]=measure_structure(images[key],np.asarray(obs.xyz),landmarks[key])
    for cur,refseg in zip(result['current']['segments'],result['reference']['segments']):
        for kind in ('xy','xyz'):
            a,b=cur[kind+'_mm'],refseg[kind+'_mm']
            cur[kind+'_ratio_to_reference']=round(a/b,3) if a is not None and b is not None and b>1 else None
    for key in ('current','reference'):
        im=Image.fromarray(images[key]);draw=ImageDraw.Draw(im);pts=result[key]['points']
        for seg in result[key]['segments']:
            a,b=(pts[n] for n in seg['endpoints'])
            if seg['xy_mm'] is not None:
                p,q=tuple(a['px']),tuple(b['px']);draw.line([p,q],fill='#00ffff',width=3)
                draw.text(((p[0]+q[0])//2,(p[1]+q[1])//2),str(seg['xy_mm'])+'mm XY',fill='yellow',stroke_width=2,stroke_fill='black')
        for name,p in pts.items():
            if p['visibility']=='VISIBLE' and p['px'] is not None:
                x,y=p['px'];draw.ellipse((x-4,y-4,x+4,y+4),fill='red')
                draw.text((x+5,y),name,fill='white',stroke_width=2,stroke_fill='black')
        im.save(Path(folder)/('structure_'+key+'.png'))
    return result
