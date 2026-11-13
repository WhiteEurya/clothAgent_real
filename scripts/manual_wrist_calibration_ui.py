"""Local manual ChArUco capture. Camera streaming and robot getters only."""
from pathlib import Path
from datetime import datetime
import sys,json,time,threading,argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np,cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_zed_realsense_charuco import pose
from cloth_agent.site_robot_check import read_arm
from cloth_agent.site_observation import load_site

HTML='''<!doctype html><meta charset="utf-8"><title>右腕标定采集</title>
<style>body{background:#141820;color:#e7edf5;font:18px system-ui;max-width:1100px;margin:24px auto}img{width:100%;border-radius:12px}button,select{padding:12px;margin:10px 8px 10px 0;font-size:18px}pre{white-space:pre-wrap;color:#8de2b0}</style>
<h2>右腕 RealSense · 手动标定采集</h2><p>仅相机预览与只读位姿记录，不控制机械臂。保持标定板固定；每次调整右臂后停稳，再点击采集。</p>
<img id="im" src="/frame.jpg"><p id="live"></p><select id="split"><option value="fit">拟合点</option><option value="holdout">独立验证点</option></select><button id="cap" onclick="capture()">采集当前点</button><pre id="result">等待采集</pre>
<script>setInterval(()=>{im.src='/frame.jpg?t='+Date.now();fetch('/status').then(x=>x.json()).then(s=>{live.textContent='角点 '+s.corners+'/24 · 已保存 '+s.saved+' 个点 · '+s.message;});},500);
async function capture(){cap.disabled=true;result.textContent='正在读取位姿并采集，请保持不动…';try{let r=await fetch('/capture',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({split:split.value})});result.textContent=JSON.stringify(await r.json(),null,2)}catch(e){result.textContent=String(e)}finally{cap.disabled=false}}</script>'''

def main():
 import pyrealsense2 as rs
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8092);p.add_argument('--output',required=True);p.add_argument('--serial',default='254522070945');p.add_argument('--arm',choices=['left','right']);a=p.parse_args()
 if a.arm is None and a.serial=='254522070945':a.arm='right'
 label={'left':'左','right':'右'}.get(a.arm,'待确认')
 page=HTML.replace('右腕',label+'腕').replace('右臂',label+'臂')
 if a.arm is None:page=page.replace('<button id="cap"','<button disabled id="cap"').replace('等待采集','安装位置待确认，目前仅预览')
 page=page.replace('<h2>','<p>相机 '+a.serial+'</p><h2>')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 lock=threading.Lock();capture_lock=threading.Lock();state={'corners':0,'saved':0,'message':'相机启动中','frame':None,'jpeg':None,'timestamp_ms':None,'received':0};stop=threading.Event()
 profile=load_site();pipe=rs.pipeline();cfg=rs.config();cfg.enable_device(a.serial);cfg.enable_stream(rs.stream.color,1280,720,rs.format.bgr8,30)
 active=pipe.start(cfg);intr=active.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
 if intr.model not in (rs.distortion.none,rs.distortion.brown_conrady) and np.any(intr.coeffs):pipe.stop();raise ValueError('Unsupported RGB distortion')
 k=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1.]])
 board=cv2.aruco.CharucoBoard((7,5),40.,30.,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000));det=cv2.aruco.CharucoDetector(board)
 (out/'session.json').write_text(json.dumps({'serial':a.serial,'mount_arm':a.arm,'robot_ip':profile['arms'][a.arm]['robot_ip'] if a.arm else None,'K':k.tolist(),'distortion':intr.coeffs,'board':{'squares':[7,5],'square_mm':40,'marker_mm':30,'dictionary':'DICT_5X5_1000'},'motion_commands':False,'automatic_merge':False},indent=2))
 def stream():
  try:
   while not stop.is_set():
    f=pipe.wait_for_frames(5000).get_color_frame();img=cv2.undistort(np.asanyarray(f.get_data()).copy(),k,np.array(intr.coeffs));stamp=f.get_timestamp();received=time.monotonic()
    corners,ids,*_=det.detectBoard(img);n=0 if ids is None else len(ids);overlay=img.copy()
    if n:cv2.aruco.drawDetectedCornersCharuco(overlay,corners.reshape(-1,1,2),ids.reshape(-1,1))
    ok,jpg=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,85])
    with lock:state.update(frame=img,jpeg=jpg.tobytes(),corners=n,timestamp_ms=stamp,received=received,message='可采集（仍需停稳）' if n>=20 else '请调整姿态，让更多棋盘进入画面')
  except Exception as exc:
   with lock:state['message']=str(exc)
 def capture(split):
  if a.arm is None:raise ValueError('请先确认相机安装位置；当前仅预览')
  before=read_arm(profile,a.arm)
  if 'joints_deg' not in before or not before.get('checks',{}).get('stationary') or not before.get('checks',{}).get('no_faults'):raise ValueError(label+'臂未静止或有故障，请停稳后再采集')
  # Manual mode is allowed. No mode changes, fault clearing, or enabling calls.
  barrier=time.monotonic();deadline=barrier+5;frames=[];last=-1
  while len(frames)<8 and time.monotonic()<deadline:
   with lock:
    if state['received']>barrier and state['timestamp_ms']!=last:
     img=state['frame'].copy();last=state['timestamp_ms'];received=state['received']
    else:img=None
   if img is None:time.sleep(.02);continue
   t,diag,overlay=pose(img,k,board)
   frames.append((img,overlay,{'camera_from_board_mm':t.tolist(),'detection':diag,'timestamp_ms':last,'host_received_monotonic':received}))
  if len(frames)<8:raise ValueError('没有足够的新帧，请重试')
  after=read_arm(profile,a.arm)
  if 'joints_deg' not in after or not after.get('checks',{}).get('stationary') or not after.get('checks',{}).get('no_faults') or np.max(np.abs(np.array(before['joints_deg'])-after['joints_deg']))>.1:raise ValueError('采集期间机械臂移动，本点未保存')
  folder=out/('point_'+datetime.now().strftime('%H%M%S_%f'));folder.mkdir()
  for i,(img,overlay,meta) in enumerate(frames):
   cv2.imwrite(str(folder/f'{i:02d}_rgb.png'),img)
   if i==0:cv2.imwrite(str(folder/'corners.png'),overlay)
  row={'split':split,'robot_before':before,'robot_after':after,'samples':[f[2] for f in frames],'image_type':'undistorted RGB','motion_commands':False}
  (folder/'observation.json').write_text(json.dumps(row,indent=2))
  with lock:state['saved']+=1
  return {'status':'SAVED','output':str(folder),'samples':8,'split':split}
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def reply(self,data,kind='application/json',code=200):
   self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path=='/':self.reply(page.encode(),'text/html; charset=utf-8')
   elif self.path.startswith('/frame.jpg'):
    with lock:data=state['jpeg']
    self.reply(data or b'', 'image/jpeg',200 if data else 503)
   elif self.path=='/status':
    with lock:d={k:state[k] for k in ('corners','saved','message')};d['frame_age_s']=time.monotonic()-state['received']
    self.reply(json.dumps(d,ensure_ascii=False).encode())
   else:self.reply(b'{}',code=404)
  def do_POST(self):
   if self.path!='/capture':self.reply(b'{}',code=404);return
   if not capture_lock.acquire(blocking=False):self.reply(b'{"error":"capture in progress"}',code=409);return
   try:
    body=json.loads(self.rfile.read(min(int(self.headers.get('Content-Length',0)),1024)));split=body.get('split','fit')
    if split not in ('fit','holdout'):raise ValueError('Invalid split')
    result=capture(split);self.reply(json.dumps(result,ensure_ascii=False).encode())
   except Exception as exc:self.reply(json.dumps({'status':'NOT_SAVED','error':str(exc)},ensure_ascii=False).encode(),code=400)
   finally:capture_lock.release()
 server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);threading.Thread(target=stream,daemon=True).start()
 print(f'MANUAL CAPTURE: http://127.0.0.1:{a.port}\nOUTPUT: {out}',flush=True)
 try:server.serve_forever()
 finally:stop.set();server.server_close();pipe.stop()
if __name__=='__main__':main()
