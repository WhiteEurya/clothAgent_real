"""Offline board-plane depth check in each camera's own optical frame."""
import argparse,json
from pathlib import Path
import cv2,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('capture',type=Path);args=p.parse_args();folder=args.capture
 meta=json.loads((folder/'report.json').read_text());cal=json.loads(Path(meta['calibration']).read_text())
 board=cv2.aruco.CharucoBoard((7,5),40.,30.,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000));results={};poses={}
 for name,key in [('zed','world_from_zed_mm'),('realsense','world_from_realsense_color_mm')]:
  im=cv2.imread(str(folder/f'{name}_rgb.png'));k=np.array(meta['zed_K'] if name=='zed' else cal['realsense_K']);corners,ids,_,_=cv2.aruco.CharucoDetector(board).detectBoard(im)
  if ids is None or len(ids)<8:raise ValueError(f'{name}: too few board corners')
  obj=board.getChessboardCorners()[ids.ravel()].astype(float);ok,rv,tv=cv2.solvePnP(obj,corners,k,None,flags=cv2.SOLVEPNP_ITERATIVE)
  if not ok:raise ValueError('PnP failed')
  r=cv2.Rodrigues(rv)[0];pred=cv2.projectPoints(obj,rv,tv,k,None)[0].reshape(-1,2);rmse=float(np.sqrt(np.mean(np.sum((pred-corners.reshape(-1,2))**2,axis=1))))
  if rmse>1.5:raise ValueError(f'{name} reprojection error {rmse}')
  T=np.array(cal[key]);world=np.load(folder/f'{name}_world_xyz_mm.npy').reshape(-1,3);cam=(world-T[:3,3])@T[:3,:3];local=(cam-tv.ravel())@r
  # Inner board area, excludes edges and background; no residual-based pruning.
  valid=np.isfinite(local).all(1)&(local[:,0]>20)&(local[:,0]<260)&(local[:,1]>20)&(local[:,1]<180)&(np.abs(local[:,2])<100)
  residual=local[valid,2]
  fit=np.linalg.lstsq(np.c_[local[valid,:2],np.ones(valid.sum())],residual,rcond=None)[0]
  results[name]={'corners':len(ids),'reprojection_rmse_px':rmse,'samples':int(valid.sum()),'signed_board_normal_residual_mm_p05_p50_p95':np.percentile(residual,[5,50,95]).tolist(),'median_absolute_plane_residual_mm':float(np.median(np.abs(residual))),'board_residual_z_ax_by_c':fit.tolist(),'note':'Positive normal follows board coordinate convention; residual against RGB-estimated board plane, not surveyed ground truth.'}
  pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=tv.ravel();poses[name]=T@pose
  uv=cv2.projectPoints(np.array([[20.,20,0],[260,20,0],[260,180,0],[20,180,0]]),rv,tv,k,None)[0].reshape(-1,2).astype(int);cv2.polylines(im,[uv],True,(0,255,0),2);cv2.imwrite(str(folder/f'{name}_board_audit.png'),im)
 delta=poses['zed'][:3,:3].T@poses['realsense'][:3,:3]
 results['cross_camera_rgb_board']={'origin_difference_mm':float(np.linalg.norm(poses['zed'][:3,3]-poses['realsense'][:3,3])),'rotation_difference_deg':float(np.degrees(np.linalg.norm(cv2.Rodrigues(delta)[0])))}
 (folder/'board_depth_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
