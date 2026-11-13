"""Round007 offline: topmost/bottommost contour crossings define a chord."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from PIL import Image,ImageDraw
from cloth_agent.contour_normal_grasp import ray_intersections
from scripts.offline_tangent_reorientation import SOURCE,ROUND,GOAL,cross,nearest
OUT=Path('reports/round007_chord_reorientation_20261108')

def crossings(a,b):
 hits=[];q=b;s=np.roll(b,-1,axis=0)-b
 for p,end in zip(a,np.roll(a,-1,axis=0)):
  r=end-p;den=r[0]*s[:,1]-r[1]*s[:,0];good=abs(den)>1e-9;v=q-p
  t=np.full(len(b),np.inf);u=t.copy()
  t[good]=(v[good,0]*s[good,1]-v[good,1]*s[good,0])/den[good]
  u[good]=(v[good,0]*r[1]-v[good,1]*r[0])/den[good]
  for h in p+t[(t>=0)&(t<=1)&(u>=0)&(u<=1),None]*r:
   if not any(np.linalg.norm(h-old)<1e-5 for old in hits):hits.append(h)
 return sorted(hits,key=lambda p:(p[1],p[0]))

def main():
 started=time.perf_counter();OUT.mkdir(exist_ok=True)
 mask=np.array(Image.open(SOURCE/'current_mask.png'))>0;goal=np.array(Image.open(SOURCE/'goal_mask.png'))>0
 _,lab,st,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8);areas=st[1:,cv2.CC_STAT_AREA].tolist();mask=lab==1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
 def contour(m):return max(cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)[:,0].astype(float)
 c=contour(mask);gc=contour(goal);cm=json.loads((ROUND/'topdown/view.json').read_text());gm=json.loads((GOAL/'topdown/view.json').read_text());scale=cm['mm_per_pixel'];center=np.array(cm['center_world_xy_mm'])
 tw=lambda p:(np.asarray(p)-320)*scale*[1,-1]+center
 tp=lambda p:(np.asarray(p)-center)/scale*[1,-1]+320
 gw=(gc-320)*gm['mm_per_pixel']*[1,-1]+gm['center_world_xy_mm'];gp=tp(gw)
 xs=crossings(c,gp)
 if len(xs)<2:raise ValueError('fewer than two contour crossings')
 A,B=xs[0],xs[-1];M=(A+B)/2;t=(B-A)/np.linalg.norm(B-A)
 outside=[]
 for p in c:
  if cv2.pointPolygonTest(gp.astype('float32').reshape(-1,1,2),tuple(p),False)<0:
   q,dist=nearest(tw(p),gw);outside.append((dist,p,q))
 dist,C,Q=max(outside,key=lambda a:a[0]);n=np.array([-t[1],t[0]])
 # Choose the half-plane opposite the protrusion, not an arbitrary PCA sign.
 if np.dot(n,C-M)>0:n=-n
 if abs(np.dot(n,C-M))<1e-6:raise ValueError('protrusion lies on chord; normal side ambiguous')
 hits=ray_intersections(c,M,n)
 if not hits:raise ValueError('no opposite contour intersection')
 E=M+hits[-1]*n;G=np.rint(E-8*n);gx,gy=G.astype(int)
 if not mask[gy,gx]:raise ValueError('inset grasp not on cloth')
 y,x=np.nonzero(mask);O=tw([x.mean(),y.mean()]);a=cross(tw(C)-O,Q-tw(C));b=cross(tw(E)-O,t*[1,-1]);bg=cross(tw(G)-O,t*[1,-1])
 if min(abs(a),abs(b),abs(bg))<1e-6:raise ValueError('ambiguous torque sign')
 direction=t if a*b>0 else -t;target=G+50/scale*direction
 report=dict(source=str(ROUND),component_areas_px=areas,all_intersections_px=[p.tolist() for p in xs],A_px=A.tolist(),B_px=B.tolist(),M_px=M.tolist(),C_px=C.tolist(),Q_px=tp(Q).tolist(),E_px=E.tolist(),G_px=G.tolist(),O_px=tp(O).tolist(),chord_length_mm=float(np.linalg.norm(B-A)*scale),unit_chord_image_xy=t.tolist(),unit_normal_image_xy=n.tolist(),normal_rule='from chord midpoint toward half-plane opposite farthest protrusion C',normal_hits_px=[(M+h*n).tolist() for h in hits],a_mm2=a,b_mm=b,b_inset_mm=bg,inset_sign_consistent=bool(b*bg>0),chosen_direction_image_xy=direction.tolist(),chosen_direction_world_xy=(direction*[1,-1]).tolist(),rotation='CCW' if a>0 else 'CW',illustrative_target_50mm_px=target.tolist(),illustrative_target_on_cloth=bool(mask[tuple(np.rint(target).astype(int)[::-1])]),robot_calls=0,model_calls=0,elapsed_s=time.perf_counter()-started,limitations=['50mm arrow illustrates sign only, not validated release point','no depth, same-arm partition, IK or collision-path verification','2D torque approximation, not physical garment rotation prediction'])
 (OUT/'geometry.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 rgb=np.array(Image.open(SOURCE/'current.png').convert('RGB'));cv2.polylines(rgb,[np.rint(gp).astype('int32')],True,(40,240,60),1)
 cv2.line(rgb,tuple(np.rint(A).astype(int)),tuple(np.rint(B).astype(int)),(255,0,255),2)
 cv2.line(rgb,tuple(np.rint(M).astype(int)),tuple(np.rint(E).astype(int)),(255,230,0),2)
 for d,col in [(direction,(0,255,255)),(-direction,(145,145,145))]:cv2.arrowedLine(rgb,tuple(G.astype(int)),tuple(np.rint(G+50/scale*d).astype(int)),col,2,tipLength=.25)
 im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
 for i,p in enumerate(xs):draw.ellipse((p[0]-2,p[1]-2,p[0]+2,p[1]+2),fill='white')
 for name,p,col,offset in [('A',A,'magenta',(7,-15)),('B',B,'magenta',(7,5)),('M',M,'yellow',(8,0)),('E',E,'yellow',(-18,-16)),('G',G,'red',(6,-12)),('C',C,'orange',(6,2)),('O',tp(O),'white',(7,2))]:
  draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=col);draw.text((p[0]+offset[0],p[1]+offset[1]),name,fill=col,stroke_width=1,stroke_fill='black')
 im.resize((1280,1280)).save(OUT/'geometry.png');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
