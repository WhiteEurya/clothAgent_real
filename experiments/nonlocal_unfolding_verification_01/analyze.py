from pathlib import Path
import json,hashlib,math
import cv2,numpy as np
from scipy.optimize import differential_evolution
from PIL import Image,ImageDraw,ImageFont
SRC=Path('reports/minimal_one_shot_6iter_20261028_165427_789294/round_002');OUT=Path(__file__).resolve().parent
before=cv2.imread(str(SRC/'before/rgb.png'));after=cv2.imread(str(SRC/'after/rgb.png'));h,w=before.shape[:2]
# Background only. Exclude garment, person, gripper, rails and image margins.
bg=np.zeros((h,w),np.uint8);bg[20:150,250:1020]=255;bg[160:650,900:1020]=255;bg[170:650,230:330]=255
sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015)
ka,da=sift.detectAndCompute(cv2.cvtColor(after,cv2.COLOR_BGR2GRAY),bg);kb,db=sift.detectAndCompute(cv2.cvtColor(before,cv2.COLOR_BGR2GRAY),bg)
matches=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.7*n.distance]
pa=np.float32([ka[m.queryIdx].pt for m in matches]);pb=np.float32([kb[m.trainIdx].pt for m in matches]);M,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=5000,confidence=.999)
assert M is not None and inliers.sum()>15
aligned=cv2.warpAffine(after,M,(w,h));errs=np.linalg.norm(cv2.transform(pa[None],M)[0]-pb,axis=1);good=inliers.ravel().astype(bool)
cv2.imwrite(str(OUT/'after_background_registered.png'),aligned)
def mask(im):
 b,g,r=cv2.split(im);a=((r>145)&(g>95)&(b>130)&(r.astype(float)>g+8)&(b.astype(float)>g+8)).astype(np.uint8)
 a[:145]=0;a[690:]=0;a[:,:340]=0;a[:,920:]=0
 n,l,stats,_=cv2.connectedComponentsWithStats(a);idx=1+np.argmax(stats[1:,4]);a=(l==idx).astype(np.uint8)
 return cv2.morphologyEx(a,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
mb,ma=mask(before),mask(aligned)
# Rigid silhouette alignment is only a diagnostic. No optical-flow/nonrigid warp.
smallb=cv2.resize(mb,(320,180),interpolation=cv2.INTER_NEAREST);smalla=cv2.resize(ma,(320,180),interpolation=cv2.INTER_NEAREST)
def trans(v,size):
 angle,dx,dy=v;T=cv2.getRotationMatrix2D((size[0]/2,size[1]/2),angle,1.);T[:,2]+=[dx,dy];return T
def iou(a,b):return float(np.logical_and(a,b).sum()/np.logical_or(a,b).sum())
def loss(v):return -iou(smallb,cv2.warpAffine(smalla,trans(v,(320,180)),(320,180),flags=cv2.INTER_NEAREST))
fit=differential_evolution(loss,[(-45,45),(-40,40),(-35,35)],seed=1,popsize=12,maxiter=120,tol=1e-7,polish=False)
v=fit.x.copy();v[1:]*=4;R=trans(v,(w,h));rigid=cv2.warpAffine(ma,R,(w,h),flags=cv2.INTER_NEAREST)
metrics={'background_registration':{'method':'SIFT ratio 0.7 + RANSAC partial affine; table-only hand-defined mask','after_to_before_matrix':M.tolist(),'matches':len(matches),'inliers':int(good.sum()),'median_inlier_residual_px':float(np.median(errs[good])),'rotation_deg':float(math.degrees(math.atan2(M[1,0],M[0,0]))),'scale':float(np.hypot(M[0,0],M[1,0])),'translation_px':M[:,2].tolist()},'silhouette_diagnostic':{'method':'threshold largest component; no manual material correspondences; best-fit 2D rotation+translation only, no scale','raw_iou':iou(mb,ma),'rigid_fitted_iou':iou(mb,rigid),'rotation_cv2_deg':float(v[0]),'translation_px':v[1:].tolist(),'before_mask_px':int(mb.sum()),'after_mask_px':int(ma.sum()),'warning':'Mask/IoU differences show shape change only; not proof of unfolding, material correspondence or layer reduction.'}}
(OUT/'registration.json').write_text(json.dumps(metrics,indent=2))
# Annotated evidence figure; pixels are observed data, not generated imagery.
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n):return ImageFont.truetype(fontpath,n)
F=font(24);SM=font(20);BIG=font(32)
B=Image.fromarray(cv2.cvtColor(before,cv2.COLOR_BGR2RGB));A=Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB))
can=Image.new('RGB',(2100,1510),'#f7f7f7');d=ImageDraw.Draw(can)
d.text((24,10),'Non-local Unfolding Verification 01 · round_002',font=BIG,fill='black')
d.text((24,55),'结论：INCONCLUSIVE｜可确认非刚性形变，但无法确认独立非目标区域的折层减少',font=F,fill='#8b4500')
boxes=[(650,200,835,390),(670,430,875,665)]
def annotated(im,action=False):
 im=im.copy();di=ImageDraw.Draw(im)
 for i,box in enumerate(boxes,1):di.rectangle(box,outline='#ff9e00',width=5);di.text((box[0]+5,box[1]+5),f'B{i}?',font=font(28),fill='#8b4500',stroke_width=2,stroke_fill='white')
 if action:
  p=(587,232);q=(430,190);di.line([p,q],fill='red',width=6);a=math.atan2(q[1]-p[1],q[0]-p[0]);di.polygon([q,(q[0]-22*math.cos(a-.45),q[1]-22*math.sin(a-.45)),(q[0]-22*math.cos(a+.45),q[1]-22*math.sin(a+.45))],fill='red')
  for pt,col,label in [(p,'lime','Pa (587,232)'),(q,'deepskyblue','Pb (430,190)')]:di.ellipse((pt[0]-8,pt[1]-8,pt[0]+8,pt[1]+8),fill=col);di.text((pt[0]-50,pt[1]-45),label,font=font(24),fill='black',stroke_width=2,stroke_fill='white')
 return im
for k,(title,im) in enumerate([('Before：原始观察',annotated(B)),('实际执行：左肩 → 左上方',annotated(B,True)),('After：仅桌面背景配准',annotated(A))]):
 x=k*700+10;d.text((x,100),title,font=F,fill='black');im.thumbnail((680,383));can.paste(im,(x,140))
d.text((20,535),'橙框 B1/B2 是待核验区域，不是已确认展开；前后同坐标框不代表材料点对应。箭头是 Pa/Pb，不是跟踪轨迹。',font=SM,fill='black')
for k,(title,box) in enumerate([('B1? 画面右肩／袖根：褶带与袖形变化',boxes[0]),('B2? 右下衣身／下摆：褶皱重排',boxes[1])]):
 x=10+700*k;d.text((x,590),title,font=SM,fill='black')
 for j,im in enumerate([B,A]):
  crop=im.crop(box);crop.thumbnail((325,300));can.paste(crop,(x+j*340,650));d.text((x+j*340,620),'Before' if j==0 else 'After (registered)',font=SM,fill='black')
cont=np.full((h,w,3),245,np.uint8)
for m,col in [(mb,(0,160,210)),(rigid,(255,130,0))]:
 cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cont,cs,-1,col,3)
im=Image.fromarray(cont).crop((300,130,940,700));im.thumbnail((600,300));can.paste(im,(1460,650))
d.text((1410,590),'轮廓刚体对齐：青 Before／橙 After',font=SM,fill='black');d.text((1410,620),f"IoU {metrics['silhouette_diagnostic']['raw_iou']:.3f} → {metrics['silhouette_diagnostic']['rigid_fitted_iou']:.3f}；残差≠展开证明",font=SM,fill='black')
d.text((20,975),'过程证据：手腕相机随机器人运动，关键阶段 B 区域不在视野内',font=F,fill='black')
frames=[('execution_images/000000.jpg','动作 0.01 s：初态'),('execution_images/000100.jpg','动作 20.02 s：主体出画'),('execution_images/000180.jpg','动作 36.03 s：主体仍出画'),('return_perception/execution_images/000074.jpg','返回 14.83 s：最终形态')]
for k,(path,title) in enumerate(frames):
 im=Image.open(SRC/path).convert('RGB');im.thumbnail((510,287));x=k*525+5;can.paste(im,(x,1050));d.text((x,1015),title,font=SM,fill='black')
d.text((20,1350),'执行成功：9/9 命令完成；XY 位移 115.95 mm；抓取姿态 Home；运动记录 lift_height=20 mm。',font=F,fill='black')
d.text((20,1390),'207 帧操作 + 75 帧返回均保留。没有独立固定机位的操作过程视频；不推断摩擦、牵力路径或层间滑动机制。',font=SM,fill='black')
d.text((20,1430),'A 的原目标还包括“左侧衣身和领口”；中央布面变化不能自动被认定为非目标收益。',font=SM,fill='black')
can.save(OUT/'comparison.png')
cv2.imwrite(str(OUT/'background_mask.png'),bg);cv2.imwrite(str(OUT/'before_mask.png'),mb*255);cv2.imwrite(str(OUT/'after_mask.png'),ma*255)
# Inventory all source files actually used, including every original recorded frame.
paths=[SRC/n for n in ['before/rgb.png','after/rgb.png','plan.json','dispatch_intent.json','execution.json','points.json','native_plan.json','motion_profile.json','states.jsonl','events.jsonl']]
for f in [SRC/'execution_images',SRC/'return_perception/execution_images']:paths.extend(sorted(f.iterdir()))
(OUT/'source_manifest.json').write_text(json.dumps({'scope':str(SRC),'files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths if p.is_file()]},indent=2))
print(json.dumps(metrics,indent=2))
