"""Local SAM2 masks from original camera RGB, projected onto the planning view.

Appearance-only masks never replace measured XYZ or execution source mapping.
"""
import atexit,json,os,selectors,subprocess,threading
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .dual_cloud_render import texture_surface

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'config/segmentation.grounded_sam2.json'
_LOCK=threading.RLock()
_PROCESS=None
_LOG=None


def close_worker():
    global _PROCESS,_LOG
    if _PROCESS is not None:
        _PROCESS.terminate()
        try:_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:_PROCESS.kill();_PROCESS.wait()
        _PROCESS=None
    if _LOG is not None:_LOG.close();_LOG=None


atexit.register(close_worker)


def segment_file(image,output,crop=None):
    global _PROCESS,_LOG
    with _LOCK:
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        if _PROCESS is None or _PROCESS.poll() is not None:
            close_worker()
            log=ROOT/'model_assets/grounded_sam2/worker.log';log.parent.mkdir(parents=True,exist_ok=True)
            _LOG=log.open('a',encoding='utf-8')
            env={**os.environ,'HF_HUB_OFFLINE':'1','PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1'}
            _PROCESS=subprocess.Popen([os.environ.get('CLOTH_SEGMENTATION_PYTHON',cfg['python']),'-m','cloth_agent.grounded_sam2_worker',str(CONFIG)],
                cwd=ROOT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=_LOG,text=True,bufsize=1)
        request=dict(image=str(Path(image).resolve()),output=str(Path(output).resolve()),crop=crop)
        try:
            _PROCESS.stdin.write(json.dumps(request)+'\n');_PROCESS.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(_PROCESS.stdout,selectors.EVENT_READ)
                if not selector.select(cfg['timeout_seconds']):raise TimeoutError('local SAM2 inference timed out')
            line=_PROCESS.stdout.readline()
            if not line:raise RuntimeError('local SAM2 worker exited; see model_assets/grounded_sam2/worker.log')
            response=json.loads(line)
            if not response['ok']:raise ValueError(response['error'])
            return np.array(Image.open(output))>0,response
        except (OSError,RuntimeError,TimeoutError,json.JSONDecodeError):
            close_worker();raise


def project_mask(mask,world,transform,intrinsics,scale,origin):
    """Same inverse-scale RGB projection as appearance rendering; no XYZ writes."""
    texture=np.repeat((np.asarray(mask,dtype=np.uint8)*255)[:,:,None],3,axis=2)
    color,valid,_=texture_surface(world,texture,np.asarray(transform),np.asarray(intrinsics),
                                 scale=scale,scale_origin=np.asarray(origin))
    return (color[:,:,0]>=128)&valid


def merge_masks(masks,owner):
    """Follow displayed surface ownership, not a union of occluded camera masks."""
    result=np.zeros(owner.shape,np.uint8)
    for i,mask in enumerate(masks):result[(owner==i)&mask]=1
    return result


def segment_view(before):
    folder=Path(before.directory)/'grounded_sam2';folder.mkdir(exist_ok=True)
    records=[]
    metadata_path=Path(before.directory)/'view.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    if metadata.get('mode') in ('RGB_FIRST_DUAL_CAMERA','RGB_FIRST_MULTI_CAMERA'):
        # Use the exact original-color mapping saved by the RGB renderer.
        # A display plane or measured XYZ must not be substituted for this map.
        owner=np.load(Path(before.directory)/'source_camera.npy')
        xy=np.load(Path(before.directory)/'source_color_xy.npy')
        visible=np.load(Path(before.directory)/'appearance_valid.npy')
        if xy.shape!=owner.shape+(2,) or visible.shape!=owner.shape:
            raise ValueError('RGB mosaic source mapping dimensions differ')
        raw=np.zeros(owner.shape,np.uint8)
        for i,record in enumerate(metadata['source_images']):
            image=Path(record['path'])
            import hashlib
            if hashlib.sha256(image.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('RGB mosaic original source image changed')
            selected=visible&(owner==i)&np.isfinite(xy).all(2)
            if not selected.any():continue
            h,w=np.asarray(Image.open(image)).shape[:2]
            y,x=np.nonzero(selected);pixels=np.rint(xy[y,x]).astype(int)
            valid=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
            if not valid.all():raise ValueError('RGB mosaic color mapping outside original image')
            lo=np.maximum(pixels.min(0)-8,0);hi=np.minimum(pixels.max(0)+9,[w,h])
            crop=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
            mask,report=segment_file(image,folder/f'camera_{i}_mask.png',crop)
            if mask.shape!=(h,w):raise ValueError('SAM2 mask dimensions differ from original color image')
            projected=np.zeros(owner.shape,np.uint8)
            projected[y,x]=mask[pixels[:,1],pixels[:,0]]
            raw[y,x]=projected[y,x]
            Image.fromarray(projected*255).save(folder/f'camera_{i}_projected_mask.png')
            records.append(dict(report,source_image=str(image),camera_id=i))
    elif hasattr(before,'sources'):
        metadata=json.loads((Path(before.directory)/'view.json').read_text(encoding='utf-8'))
        zpath=Path(before.directory)/'appearance_display_z_mm.npy'
        if not zpath.exists():raise ValueError('grounded_sam2 requires surface_rgb dual topdown')
        z=np.load(zpath);y,x=np.indices(z.shape)
        center=metadata['center_world_xy_mm'];span=metadata['span_mm']
        world=np.stack([center[0]+(x-320)*span/520,center[1]-(y-320)*span/520,z],axis=2)
        owner=np.load(Path(before.directory)/'appearance_source_camera.npy')
        cfg=before.source.meta['dual_cloud']['config'];projected=[]
        for i,source in enumerate(before.sources):
            image=source.image if i==0 else source.directory/'color_rgb.png'
            T=np.asarray(source.meta['world_from_camera_mm'] if i==0 else cfg['calibration']['world_from_realsense_color_mm'])
            K=np.asarray(source.meta['intrinsics'] if i==0 else cfg['calibration']['realsense_K'])
            scale=cfg['depth_scale']['zed' if i==0 else 'realsense'];origin=np.asarray(source.meta['world_from_camera_mm'])[:3,3]
            # Table-sized RGB crop, derived from existing table display, not a hand-picked garment box.
            image_size=Image.open(image).size
            _,valid,uv=texture_surface(world,np.zeros((*image_size[::-1],3),np.uint8),T,K,scale=scale,scale_origin=origin)
            if not valid.any():raise ValueError('no visible table for original RGB segmentation')
            points=uv[valid];lo=np.maximum(np.floor(points.min(0)).astype(int)-8,0);hi=np.minimum(np.ceil(points.max(0)).astype(int)+9,image_size)
            crop=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]
            mask,report=segment_file(image,folder/f'camera_{i}_mask.png',crop)
            records.append(dict(report,source_image=str(image)))
            projected.append(project_mask(mask,world,T,K,scale,origin))
            Image.fromarray(projected[-1].astype('uint8')*255).save(folder/f'camera_{i}_projected_mask.png')
        raw=merge_masks(projected,owner)
    else:
        source=getattr(before,'source',before)
        mask,report=segment_file(source.image,folder/'camera_0_mask.png')
        records.append(dict(report,source_image=str(source.image)))
        if source is before:raw=mask.astype('uint8')
        else:
            xy=before.source_xy;valid=(xy[:,:,0]>=0)&(xy[:,:,1]>=0)&(xy[:,:,0]<mask.shape[1])&(xy[:,:,1]<mask.shape[0])
            raw=np.zeros(xy.shape[:2],np.uint8);raw[valid]=mask[xy[:,:,1][valid],xy[:,:,0][valid]]
    Image.fromarray(raw*255).save(folder/'projected_raw_mask.png')
    # Close only one-pixel projection gaps; measured depth remains unchanged.
    result=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(result,8)
    if n<2:raise ValueError('no clothing survives projection into current planning view')
    result=(labels==1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
    Image.fromarray(result*255).save(folder/'mask.png')
    rgb=np.array(Image.open(before.image).convert('RGB'));contours,_=cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb,contours,-1,(0,255,70),1);Image.fromarray(rgb).save(folder/'boundary_overlay.png')
    report=dict(method='grounded_sam2_original_rgb_projected',profile='grounded_sam2',sources=records,
        area_px=int(result.sum()),depth_modified=False,projection='same displayed camera/surface; original measured motion depth unchanged',
        mask_processing='3x3 close then largest component',source_image=str(before.image))
    if hasattr(before,'unknown_mask'):
        report['silhouette_cells_without_measured_depth']=int(((result>0)&before.unknown_mask).sum())
    (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return result,report
