"""Static-scene camera-only fusion preview; no robot imports or commands."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,time
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.zed_rgbd import ZEDRGBD
from cloth_agent.pixel_mapping_visualization import PixelMappingView
from reconstruct_level_cloud import save_ply


def transform(x,t):
    t=np.asarray(t);return x@t[:3,:3].T+t[:3,3]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rs-depth-size',nargs=2,type=int,default=[848,480]);p.add_argument('--zed-resolution',choices=['HD720','HD1080'],default='HD720');p.add_argument('--zed-depth-mode',choices=['ULTRA','NEURAL','NEURAL_LIGHT'],default='ULTRA');p.add_argument('--calibration',default='config/calibration/zed_realsense_20261105/extrinsics.json');p.add_argument('--scene',choices=['garment','calibration'],default='garment');p.add_argument('--depth-frames',type=int,default=1);args=p.parse_args()
    if not 1<=args.depth_frames<=120:p.error('--depth-frames must be 1..120')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cal=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    calibration_snapshot=out/'calibration_used.json'
    calibration_snapshot.write_text(json.dumps(cal,indent=2),encoding='utf-8')
    roi=json.loads(Path('config/topdown.unfold_zed.json').read_text())
    spec=SimpleNamespace(serial=cal['zed_serial'],depth_mode=args.zed_depth_mode,confidence_threshold=50,color_exposure=None,color_white_balance=None,depth_exposure_us=None,depth_gain=None)
    zwid,zhei,zfps=(1280,720,30) if args.zed_resolution=='HD720' else (1920,1080,15)
    cam=ZEDRGBD(spec,zwid,zhei,zfps)
    try:
        cam.start()
        for _ in range(35): zr,zd=cam.read()
        zstack=[]
        for _ in range(args.depth_frames):
            zr,zd=cam.read();zstack.append(np.array(zd,dtype=np.float32,copy=True))
        np.save(out/'zed_depth_frames_m.npy',np.stack(zstack))
        kz=cam.intrinsics.copy();zt=cam.timestamp_ns
        np.save(out/'zed_confidence.npy',cam.confidence)
        sdk_xyz=cam.sl.Mat()
        cam._check(cam.camera.retrieve_measure(sdk_xyz,cam.sl.MEASURE.XYZ),'SDK XYZ audit')
        zed_sdk_xyz=np.array(sdk_xyz.get_data()[...,:3],copy=True)*1000
    finally: cam.stop()
    yy,xx=np.indices(zd.shape)
    zc=np.stack(((xx-kz[0,2])/kz[0,0]*zd,(yy-kz[1,2])/kz[1,1]*zd,zd),axis=-1)*1000
    both=np.isfinite(zc).all(2)&np.isfinite(zed_sdk_xyz).all(2)
    zed_sdk_xyz_max_error_mm=float(np.max(np.linalg.norm(zc[both]-zed_sdk_xyz[both],axis=1)))
    if zed_sdk_xyz_max_error_mm>0.1:raise ValueError('ZED deprojection disagrees with SDK XYZ')
    zw=transform(zc,cal['world_from_zed_mm'])
    Image.fromarray(zr).save(out/'zed_rgb.png');np.save(out/'zed_depth_m.npy',zd)
    import pyrealsense2 as rs
    pipe=rs.pipeline();cfg=rs.config();cfg.enable_device(cal['realsense_serial']);cfg.enable_stream(rs.stream.color,1280,720,rs.format.rgb8,30);cfg.enable_stream(rs.stream.depth,*args.rs_depth_size,rs.format.z16,30)
    started=False
    try:
        profile=pipe.start(cfg);started=True
        for _ in range(60): frames=pipe.wait_for_frames(5000)
        scale=profile.get_device().first_depth_sensor().get_depth_scale()
        rstack=[]
        for _ in range(args.depth_frames):
            frames=pipe.wait_for_frames(5000)
            rstack.append(np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)*scale)
        np.save(out/'realsense_depth_frames_m.npy',np.stack(rstack))
        depth=frames.get_depth_frame();color=frames.get_color_frame();rr=np.asanyarray(color.get_data()).copy()
        pc=rs.pointcloud();pc.map_to(color);points=pc.calculate(depth)
        rd=np.asanyarray(depth.get_data()).copy();scale=profile.get_device().first_depth_sensor().get_depth_scale()
        vertices=np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1,3).copy()*1000
        uv=np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1,2).copy()
        ext=depth.profile.get_extrinsics_to(color.profile)
        # librealsense stores rotation in column-major order.
        d2c=np.eye(4);d2c[:3,:3]=np.array(ext.rotation).reshape(3,3,order='F');d2c[:3,3]=np.array(ext.translation)*1000
        test_ids=np.flatnonzero(np.isfinite(vertices).all(1)&(vertices[:,2]>150))[::1000]
        manual=transform(vertices[test_ids],d2c)
        sdk=np.array([rs.rs2_transform_point_to_point(ext,(vertices[i]/1000).tolist()) for i in test_ids])*1000
        sdk_transform_max_error_mm=float(np.max(np.linalg.norm(manual-sdk,axis=1)))
        if sdk_transform_max_error_mm>0.01:raise ValueError('Depth-to-color transform disagrees with SDK')
        color_intr=color.profile.as_video_stream_profile().get_intrinsics()
        rs_color_intrinsics=dict(fx=color_intr.fx,fy=color_intr.fy,ppx=color_intr.ppx,ppy=color_intr.ppy,coeffs=color_intr.coeffs,model=str(color_intr.model))
        rw=transform(transform(vertices,d2c),cal['world_from_realsense_color_mm'])
        pixel=np.rint(uv*np.array([rr.shape[1],rr.shape[0]])).astype(int)
        valid=(vertices[:,2]>150)&(vertices[:,2]<2000)&np.isfinite(uv).all(1)&(pixel[:,0]>=0)&(pixel[:,0]<rr.shape[1])&(pixel[:,1]>=0)&(pixel[:,1]<rr.shape[0])
        rc=np.zeros((len(vertices),3),np.uint8);rc[valid]=rr[pixel[valid,1],pixel[valid,0]];rw[~valid]=np.nan
        rt=color.get_timestamp()
    finally:
        if started:pipe.stop()
    Image.fromarray(rr).save(out/'realsense_rgb.png');np.save(out/'realsense_native_depth_m.npy',rd*scale)
    np.save(out/'zed_world_xyz_mm.npy',zw);np.save(out/'realsense_world_xyz_mm.npy',rw)
    import cv2
    import yaml
    roi_k=np.array(yaml.safe_load(Path('config/calibration/zed_dual_xarm7_20261103/right/intrinsics_left.yaml').read_text())['K'])
    mask=np.zeros((720,1280),np.uint8);cv2.fillPoly(mask,[np.array(roi['table_polygon_px'],np.int32)],1)
    def crop(world):
        flat=world.reshape(-1,3);camxyz=(flat-np.array(cal['world_from_zed_mm'])[:3,3])@np.array(cal['world_from_zed_mm'])[:3,:3]
        good=np.isfinite(flat).all(1)&(camxyz[:,2]>0)&(flat[:,2]>=-20)&(flat[:,2]<=400)
        ids=np.flatnonzero(good);uv=camxyz[ids]@roi_k.T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<1280)&(uv[:,1]>=0)&(uv[:,1]<720);good[:]=False
        ii=ids[inside];pix=uv[inside];good[ii]=mask[pix[:,1],pix[:,0]]>0
        return good
    zm=crop(zw);rm=crop(rw)
    clouds=[zw.reshape(-1,3)[zm],rw[rm]];colors=[zr.reshape(-1,3)[zm],rc[rm]]
    clouds.append(np.concatenate(clouds));colors.append(np.concatenate(colors))
    views=[]
    for name,x,c in zip(['zed','realsense','fused'],clouds,colors):
        save_ply(out/f'{name}.ply',x,c)
        view=PixelMappingView(c[None,:,:],x[None,:,:],np.ones((1,len(x)),bool),show_xy_grid=False,max_points=None)
        views.append(view)
    common=views[-1];report={'artifact_role':'UNVALIDATED_POINT_CLOUD_OVERLAY','fusion_method':'Concatenate measured world points; top-down chooses highest measured Z; no surface reconciliation or depth bias correction','capture':'sequential static scene, no hardware synchronization','depth_mode':f'ZED {args.zed_depth_mode}; RealSense native depth without hole filling','units':'mm','crop':'existing ZED table polygon; display-only world Z -20..400 mm','robot_connections':0,'depth_frames':args.depth_frames,'realsense_depth_size_wh':args.rs_depth_size,'zed_timestamp_ns':zt,'realsense_timestamp_ms':rt,'realsense_depth_to_color_mm':d2c.tolist(),'sdk_transform_max_error_mm':sdk_transform_max_error_mm,'realsense_color_intrinsics':rs_color_intrinsics,'zed_K':kz.tolist(),'zed_resolution':args.zed_resolution,'zed_sdk_xyz_max_error_mm':zed_sdk_xyz_max_error_mm,'roi_reference_K':roi_k.tolist(),'calibration':str(calibration_snapshot),'calibration_source':args.calibration,'counts':{},'topdown_center_mm':common.center.tolist(),'topdown_span_mm':common.span}
    canvas=Image.new('RGB',(1920,680),'#181e27');draw=ImageDraw.Draw(canvas)
    grids=[]
    for i,(name,view) in enumerate(zip(['zed','realsense','fused'],views)):
        view.center=common.center;view.span=common.span;view._xy_image();view.measured_image.save(out/f'{name}_topdown.png')
        np.save(out/f'{name}_topdown_xyz_mm.npy',view.projected_xyz)
        canvas.paste(view.measured_image,(i*640,40));draw.text((i*640+20,12),name.upper()+' | same world XY scale | no hole filling',fill='white')
        occupied=np.isfinite(view.projected_xyz).all(2);grids.append(occupied);report['counts'][name]={'points':len(clouds[i]),'occupied_topdown_cells':int(occupied.sum())}
    report['cells_added_to_zed']=int((grids[1]&~grids[0]).sum())
    overlap=grids[0]&grids[1];dz=views[1].projected_xyz[:,:,2][overlap]-views[0].projected_xyz[:,:,2][overlap]
    report['overlap_height_difference_mm']={'cells':len(dz),'median_signed':float(np.median(dz)),'median_absolute':float(np.median(np.abs(dz))),'p95_absolute':float(np.percentile(np.abs(dz),95)),'note':'Same raster cell, potentially different surfaces/occlusions; not calibration error alone.'}
    canvas.save(out/'comparison.png')
    raw=Image.new('RGB',(1280,400),'#181e27')
    for i,img in enumerate([zr,rr]):raw.paste(Image.fromarray(img).resize((640,360)),(i*640,40))
    d=ImageDraw.Draw(raw);d.text((20,10),'ZED RGB',fill='white');d.text((660,10),'REALSENSE RGB',fill='white');raw.save(out/'raw_comparison.jpg')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Keep raw geometry diagnostics, but present the same RGB renderer as unfold.
    from cloth_agent.dual_cloud_preview_appearance import render_saved_preview
    fusion=json.loads(Path('config/fusion.unfold_dual.json').read_text(encoding='utf-8'))
    image=render_saved_preview(out,fusion,roi,scene=args.scene)
    report['appearance_preview']=str(image)
    report['raw_fused_topdown_role']='geometry diagnostic; unharmonized camera colors'
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    print('FUSED RGB: '+str(image),flush=True)

if __name__=='__main__':main()
