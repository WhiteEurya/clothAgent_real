from pathlib import Path
import sys,json,shutil,hashlib
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/e353_body_alignment_20261029';R.mkdir(exist_ok=True)
S=ROOT/'reports/controlled_ablation_20261026/robot_v2/runs/episode_0007/round_004'
for n,d in [('before/rgb.png','before.png'),('after/rgb.png','after.png')]:
 if not (R/d).exists():shutil.copyfile(S/n,R/d)
a=cv2.imread(str(R/'before.png'));b=cv2.imread(str(R/'after.png'))
# Chest print anchors only, selected from Before geometry before any model inference.
roi=np.zeros(a.shape[:2],np.uint8);roi[300:480,600:825]=255
sift=cv2.SIFT_create();ka,da=sift.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),roi);kb,db=sift.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
m=[x for x,y in cv2.BFMatcher().knnMatch(da,db,k=2) if x.distance<.7*y.distance]
x=np.float64([ka[v.queryIdx].pt for v in m]);y=np.float64([kb[v.trainIdx].pt for v in m]);est,ins=cv2.estimateAffinePartial2D(x,y,method=cv2.RANSAC,ransacReprojThreshold=3)
assert est is not None and int(ins.sum())>=6
ok=ins[:,0].astype(bool);xx=x[ok];yy=y[ok];xc=xx.mean(0);yc=yy.mean(0);u,sv,vt=np.linalg.svd((xx-xc).T@(yy-yc));rot=vt.T@u.T
if np.linalg.det(rot)<0:vt[-1]*=-1;rot=vt.T@u.T
trans=yc-rot@xc;M=np.column_stack((rot,trans));warp=cv2.warpAffine(a,M,(a.shape[1],a.shape[0]));cv2.imwrite(str(R/'before_aligned.png'),warp)
blend=cv2.addWeighted(warp,.5,b,.5,0);cv2.imwrite(str(R/'aligned_overlay.png'),blend)
# Edge display distinguishes projected outline change from material tracking.
def mask(im):
 hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=cv2.inRange(hsv,np.array([120,20,55]),np.array([170,200,255]));n,l,stats,_=cv2.connectedComponentsWithStats(m)
 return np.uint8(l==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA])))*255
ma=mask(a);mb=mask(b);mw=cv2.warpAffine(ma,M,(a.shape[1],a.shape[0]),flags=cv2.INTER_NEAREST)
vis=b.copy();ca=cv2.findContours(mw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0];cb=cv2.findContours(mb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0];cv2.drawContours(vis,ca,-1,(255,255,0),2);cv2.drawContours(vis,cb,-1,(255,0,255),2)
cv2.putText(vis,'CYAN: aligned before | MAGENTA: after',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);cv2.imwrite(str(R/'aligned_contours.png'),vis)
draw=b.copy()
for q,z in zip(xx,yy):
 p=M[:,:2]@q+M[:,2];cv2.circle(draw,tuple(z.astype(int)),3,(0,255,0),1);cv2.line(draw,tuple(p.astype(int)),tuple(z.astype(int)),(0,0,255),1)
cv2.imwrite(str(R/'body_anchor_fit.png'),draw)
metric={'method':'SIFT chest-print matches; RANSAC similarity inliers; rigid no-scale SVD refit','before_anchor_roi_xyxy':[600,300,825,480],'matches':len(m),'inliers':len(xx),'rigid_before_to_after':M.tolist(),'image_rotation_deg':float(np.degrees(np.arctan2(rot[1,0],rot[0,0]))),'chest_centroid_displacement_px':(yc-xc).tolist(),'median_anchor_residual_px':float(np.median(np.linalg.norm(xx@rot.T+trans-yy,axis=1))),'p95_anchor_residual_px':float(np.quantile(np.linalg.norm(xx@rot.T+trans-yy,axis=1),.95)),'similarity_scale_diagnostic':float(np.linalg.norm(est[:,0])),'limitations':['Alignment anchored to chest print, not complete rigid torso truth.','No material tracking outside chest anchors.','New/lost contour pixels may be unocclusion or folds, not displacement vectors.','RGB projection is not a metric 3D deformation reconstruction.']}
(R/'alignment.json').write_text(json.dumps(metric,ensure_ascii=False,indent=2))
for n in ['plan.json','geometry.json','execution.json','motion.py']:shutil.copyfile(S/n,R/n)
manifest={'source':str(S),'inputs_sha256':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['before.png','after.png','plan.json','geometry.json','execution.json']},'chat_history_included':False,'previous_analysis_included':False,'scope':'Raw E353 plus deterministic alignment; historical planner reason supplied only as recorded intention, not true anatomy.'}
(R/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'));import engine;engine.OUT=R
st={'type':'string'};fields={k:st for k in ['recorded_intention','visible_structure_before','visible_structure_after','body_alignment_interpretation','common_motion','relative_deformation','what_actually_changed','mechanism_1','mechanism_2','mechanism_3_optional','discriminating_evidence_needed','conclusion']};schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
payload={'task':'E353中机器人的意图移动是什么，实际上衣服结构是什么？先对齐操作前后的衣身主体，分析哪些区域发生整体运动、哪些区域发生相对形变。解释袖子为什么显露，提出至少两种能够产生该现象的物理机制。不要先归纳抓点与运动方向共同规律，先解释变化，再讨论机制。','image_order':['raw Before','raw After','rigid aligned Before blended 50% with After','aligned contours: cyan Before, magenta After','chest anchor matches in After'],'historical_planner_intention_not_ground_truth':json.loads((R/'plan.json').read_text()),'executed_parameters':json.loads((R/'geometry.json').read_text()),'controller_record_not_grasp_proof':json.loads((R/'execution.json').read_text()),'alignment':metric,'rules':['Independently infer visible anatomy and semantic vs image sides; never assume planner labels correct.','Only two endpoint observations: causal mechanisms are hypotheses, not observed hidden topology.','Do not equate contour visibility with material point displacement.','Do not give next action, skill, grasp-side selector or general action direction rule.','For each mechanism state sequence, supporting facts, unresolved assumptions, and how to distinguish it.']}
r=engine.call('independent_structure_analysis','你在一个全新隔离会话中进行机器人操作的物理分析。只使用附带原始记录、图片和计算结果，不读取其他文件或调用工具。中文回答。严格区分记录的意图、可见结构、测量和机制假设。不得将意图当作抓中部位的证明。',payload,[R/n for n in ['before.png','after.png','aligned_overlay.png','aligned_contours.png','body_anchor_fit.png']],schema,'independent_e353_analysis',['E353'])
(R/'analysis.json').write_text(json.dumps(engine.require(r),ensure_ascii=False,indent=2));print(json.dumps(metric,ensure_ascii=False,indent=2))
