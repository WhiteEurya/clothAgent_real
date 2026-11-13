"""Camera-only fixed ZED/RealSense board alignment; translations in millimetres."""
from pathlib import Path
import argparse
import json
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth_agent.dual_camera_tag import fit_camera_transform, compare_tag, summarize


def pose(image, k, board):
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids) < 20:
        raise ValueError('Need at least 20/24 board corners')
    obj = board.getChessboardCorners()[ids.ravel()].astype(np.float64)
    ok, rvs, tvs, _ = cv2.solvePnPGeneric(obj, corners, k, None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rv, tv in zip(rvs, tvs):
        rot = cv2.Rodrigues(rv)[0]
        if not np.all((obj @ rot.T + tv.reshape(3))[:, 2] > 0):
            continue
        projected = cv2.projectPoints(obj, rv, tv, k, None)[0]
        err = float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2, axis=1))))
        t = np.eye(4); t[:3,:3] = rot; t[:3,3] = tv.ravel()
        candidates.append((err, t))
    candidates.sort(key=lambda x: x[0])
    if not ok or not candidates or candidates[0][0] > 1.5:
        raise ValueError(f'Board reprojection error exceeds 1.5 pixels: {[c[0] for c in candidates]}')
    angle = 0. if len(candidates)<2 else np.degrees(np.linalg.norm(cv2.Rodrigues(candidates[0][1][:3,:3].T @ candidates[1][1][:3,:3])[0]))
    if len(candidates)>1 and angle > 5 and candidates[1][0] < candidates[0][0]+0.2:
        raise ValueError('Ambiguous planar board pose')
    preview = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(preview, corners.reshape(-1,1,2), ids.reshape(-1,1))
    cv2.drawFrameAxes(preview, k, None, cv2.Rodrigues(candidates[0][1][:3,:3])[0], candidates[0][1][:3,3], 80)
    return candidates[0][1], dict(corner_ids=ids.ravel().tolist(), corners=corners.reshape(-1,2).tolist(), reprojection_rmse_px=candidates[0][0]), preview


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--reference', help='Previous calibration.json to independently compare at a new board placement')
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    import pyrealsense2 as rs
    import pyzed.sl as sl
    board=cv2.aruco.CharucoBoard((7,5),40.,30.,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000))
    pipe=rs.pipeline(); cfg=rs.config(); cfg.enable_device('254522070945'); cfg.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
    zed=sl.Camera(); started=False
    rows=[]; pairs=[]; rejected=[]
    try:
        profile=pipe.start(cfg); started=True
        intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        if intr.model not in (rs.distortion.none,rs.distortion.brown_conrady) and np.any(np.asarray(intr.coeffs) != 0):
            raise ValueError(f'Unsupported nonzero color distortion: {intr.model}, {intr.coeffs}')
        kr=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
        init=sl.InitParameters(); init.set_from_serial_number(11483267); init.camera_resolution=sl.RESOLUTION.HD1080; init.camera_fps=15; init.depth_mode=sl.DEPTH_MODE.NONE; init.coordinate_system=sl.COORDINATE_SYSTEM.IMAGE
        status=zed.open(init)
        if status!=sl.ERROR_CODE.SUCCESS: raise RuntimeError(str(status))
        cam=zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        kz=np.array([[cam.fx,0,cam.cx],[0,cam.fy,cam.cy],[0,0,1.]])
        mat=sl.Mat()
        for _ in range(30): pipe.wait_for_frames(5000); zed.grab()
        deadline=time.monotonic()+45
        while len(pairs)<30 and time.monotonic()<deadline:
            f=pipe.wait_for_frames(5000).get_color_frame()
            if zed.grab()!=sl.ERROR_CODE.SUCCESS: continue
            if zed.retrieve_image(mat,sl.VIEW.LEFT)!=sl.ERROR_CODE.SUCCESS: continue
            zi=mat.get_data()[:,:,:3].copy()
            ri=cv2.undistort(np.asanyarray(f.get_data()).copy(),kr,np.array(intr.coeffs))
            try:
                a,ar,ap=pose(zi,kz,board); b,br,bp=pose(ri,kr,board)
            except ValueError as e:
                rejected.append(str(e))
                if len(rejected)==1:
                    cv2.imwrite(str(out/'rejected_zed.png'),zi); cv2.imwrite(str(out/'rejected_realsense.png'),ri)
                    print(str(e),flush=True)
                continue
            idx=len(pairs); pairs.append((a,b))
            rows.append(dict(zed=ar,realsense=br,zed_from_board_mm=a.tolist(),realsense_color_from_board_mm=b.tolist(),realsense_timestamp_ms=f.get_timestamp(),zed_timestamp_ns=zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()))
            if idx in (0,29):
                for label,img in [('zed',zi),('realsense_rectified',ri),('zed_corners',ap),('realsense_corners',bp)]: cv2.imwrite(str(out/f'{idx:03d}_{label}.png'),img)
            time.sleep(.1)
        if len(pairs)<10: raise RuntimeError(f'Only {len(pairs)} good image pairs: {rejected[-5:]}')
        transform=fit_camera_transform(pairs)
        world=np.array(json.loads(Path('config/dual_arm.zed_dual_xarm7_20261103.json').read_text())['camera']['world_from_camera_mm'])
        report=dict(status='SINGLE_PLACEMENT_FIT',board=dict(squares=[7,5],square_mm=40,marker_mm=30,ids=list(range(17))),translation_units='mm',zed_serial='11483267',realsense_serial='254522070945',zed_frame='rectified left optical IMAGE',realsense_frame='color optical; NOT depth optical',zed_K=kz.tolist(),realsense_K=kr.tolist(),realsense_original_distortion=intr.coeffs,zed_from_realsense_color_mm=transform.tolist(),world_from_realsense_color_mm=(world@transform).tolist(),world_from_zed_mm=world.tolist(),pairs=rows,rejected=rejected,fit_repeatability=summarize([compare_tag(a,b,transform) for a,b in pairs]),limitation='Same stationary board placement; repeatability is not workspace accuracy. Inherits existing ZED world calibration error. Cameras are not hardware synchronized; board must remain stationary.')
        if args.reference:
            ref=json.loads(Path(args.reference).read_text(encoding='utf-8'))
            report['reference_validation']=summarize([compare_tag(a,b,np.array(ref['zed_from_realsense_color_mm'])) for a,b in pairs])
        (out/'calibration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:report[k] for k in ('status','fit_repeatability')}),flush=True)
        if args.reference: print(json.dumps(report['reference_validation']),flush=True)
        print(str(out),flush=True)
    finally:
        zed.close()
        if started: pipe.stop()

if __name__=='__main__': main()
