"""One ZED connection shared by continuous RGB recording and fresh RGB-D capture.

No robot access. Camera calls run in one worker; observations receive distinct
fresh frames, retaining the existing temporal-median observation code.
"""
from pathlib import Path
import json
import shutil
import subprocess
import threading
import time
import logging
import numpy as np
from .perception import PerceptionError, create_rgbd_camera


class UnfoldVideo:
    def __init__(self,config,directory,report,*,fps=10,camera=None):
        active=[s for s in config.cameras if s.label in config.active_camera_labels]
        if len(active)!=1 or (active[0].backend!='zed' and not (active[0].backend=='realsense' and camera is not None)):
            raise PerceptionError('unfold recording requires the existing single ZED primary')
        self.config=config;self.spec=active[0];self.fps=fps
        self.camera=camera or create_rgbd_camera(self.spec,config.width,config.height,config.fps)
        self.directory=Path(directory);self.report=Path(report);self.path=self.directory/('left_wrist_rgb.mp4' if self.spec.backend=='realsense' else 'zed_rgb.mp4')
        self.condition=threading.Condition();self.stop_event=threading.Event()
        self.seq=0;self.consumed=0;self.latest=None;self.error=None
        self.thread=None;self.encoder=None;self.video_error=None;self.frames=0;self.closed=False
        self.intrinsics=None;self.started_at=None;self.camera_reconnects=0;self.max_camera_reconnects=3
        self._camera_failure_streak=0
    def _manifest(self,status):
        data=dict(status=status,video=str(self.path.resolve()),camera_serial=self.spec.serial,
            fps=self.fps,frames=self.frames,started_unix_s=self.started_at,camera_backend=self.spec.backend,
            video_error=self.video_error,camera_error=self.error,
            source=(f'continuous {self.spec.backend} original RGB; recording only, depth disabled'
                    if getattr(self.camera,'rgb_only',False) else
                    f'continuous {self.spec.backend} original RGB; shared RGB-D camera connection'),
            resolution=[self.config.width,self.config.height])
        data.update(camera_reconnects=self.camera_reconnects,
                    max_camera_reconnects=self.max_camera_reconnects)
        for path in (self.directory/'recording.json',self.report/'video_recording.json'):
            path.write_text(json.dumps(data,indent=2),encoding='utf-8')
    def start(self):
        self.directory.mkdir(parents=True,exist_ok=False)
        if shutil.disk_usage(self.directory).free<1024**3:
            raise RuntimeError('video disk has less than 1 GiB free')
        self.started_at=time.time()
        try:
            self.camera.start();self.intrinsics=self.camera.intrinsics
            self.log=open(self.directory/'ffmpeg.log','wb')
            self.encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n',
                '-f','rawvideo','-pix_fmt','rgb24','-s',f'{self.config.width}x{self.config.height}',
                '-r',str(self.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','ultrafast',
                '-crf','23','-pix_fmt','yuv420p','-movflags','+frag_keyframe+empty_moov',str(self.path)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
            self._manifest('RECORDING')
            self.thread=threading.Thread(target=self._loop,name='unfold-zed-video',daemon=True);self.thread.start()
            self.read()  # Confirm a fresh camera frame before any startup motion.
        except BaseException:
            self.close();raise
    def _loop(self):
        beginning=time.monotonic();last_disk=beginning
        try:
            with open(self.directory/'frame_timestamps.jsonl','w',encoding='utf-8') as timestamps:
                while not self.stop_event.is_set():
                    try:
                        rgb,depth=self.camera.read()
                    except Exception as exc:
                        if self.stop_event.is_set():break
                        self._camera_failure_streak+=1
                        if self._camera_failure_streak>self.max_camera_reconnects:
                            raise PerceptionError(
                                f'camera reconnect exhausted after {self.max_camera_reconnects} attempts: {exc}') from exc
                        last=exc;recovered=False
                        for attempt in range(1,self.max_camera_reconnects-self._camera_failure_streak+2):
                            try:
                                try:self.camera.stop()
                                except Exception:pass
                                time.sleep(float(attempt))
                                self.camera.start();self.intrinsics=self.camera.intrinsics
                            except Exception as restart_error:
                                last=restart_error
                                continue
                            self.camera_reconnects+=1;recovered=True
                            logging.warning('Camera stream recovered after read failure: %s',exc)
                            self._manifest('RECORDING')
                            break
                        if not recovered:
                            raise PerceptionError(
                                f'camera reconnect exhausted after {self.max_camera_reconnects} attempts: {last}') from last
                        continue
                    self._camera_failure_streak=0
                    now=time.monotonic()
                    with self.condition:
                        self.latest=(rgb.copy(),depth.copy());self.latest_native_frames=getattr(self.camera,'native_frames',None);self.seq+=1;self.condition.notify_all()
                    if self.video_error is not None:continue
                    try:
                        if now-last_disk>=10:
                            if shutil.disk_usage(self.directory).free<1024**3:raise OSError('less than 1 GiB disk space remains')
                            last_disk=now
                        # CFR timeline follows wall time; repeat the newest frame
                        # across short acquisition gaps, recorded explicitly below.
                        due=int((now-beginning)*self.fps)+1-self.frames
                        for _ in range(max(0,due)):
                            self.encoder.stdin.write(np.ascontiguousarray(rgb).tobytes())
                            timestamps.write(json.dumps(dict(frame=self.frames,elapsed_s=now-beginning,
                                camera_timestamp_ns=getattr(self.camera,'timestamp_ns',None),source_sequence=self.seq))+'\n')
                            self.frames+=1
                        timestamps.flush()
                    except Exception as exc:
                        self.video_error=f'{type(exc).__name__}: {exc}'
                        logging.error('Video recording failed; RGB-D observation continues: %s',self.video_error)
                        self._manifest('VIDEO_FAILED')
        except Exception as exc:
            with self.condition:
                self.error=f'{type(exc).__name__}: {exc}';self.condition.notify_all()
    def read(self):
        with self.condition:
            ready=self.condition.wait_for(lambda:self.seq>self.consumed or self.error or self.stop_event.is_set(),timeout=10)
            if self.error:raise PerceptionError(self.error)
            if not ready or self.stop_event.is_set():raise PerceptionError('continuous ZED capture stopped or timed out')
            self.consumed=self.seq
            return tuple(v.copy() for v in self.latest)
    def camera_factory(self,spec,width,height,fps):
        if spec!=self.spec or (width,height,fps)!=(self.config.width,self.config.height,self.config.fps):
            raise PerceptionError('recording/observation camera configuration mismatch')
        owner=self
        class BorrowedCamera:
            def __init__(self):self.spec=owner.spec;self.intrinsics=owner.intrinsics
            def start(self):pass
            def read(self):return owner.read()
            def stop(self):pass  # Camera lifetime belongs to the whole run.
        return BorrowedCamera()
    def close(self):
        if self.closed:return
        self.closed=True;self.stop_event.set()
        with self.condition:self.condition.notify_all()
        if self.thread:self.thread.join(timeout=15)
        if self.thread and self.thread.is_alive():
            self.error='camera worker did not stop within 15 s'
            self._manifest('STOP_TIMEOUT');return
        try:
            if self.encoder:
                try:self.encoder.stdin.close()
                except (OSError,BrokenPipeError):pass
                try:code=self.encoder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.encoder.kill();code=self.encoder.wait();self.video_error='encoder stop timeout'
                if code and not self.video_error:self.video_error=f'ffmpeg exit {code}'
        finally:
            self.camera.stop()
            if hasattr(self,'log'):self.log.close()
            self._manifest('FAILED' if self.error or self.video_error or not self.frames else 'COMPLETED')
            if self.frames and self.path.exists():
                try:
                    result=retain_32x_video(self.directory,self.report)
                    self.path=Path(result['video'])
                    print(f"VIDEO 32x: {self.path}",flush=True)
                except Exception as exc:
                    logging.error('32x conversion failed; original video retained: %s',exc)
                    for manifest in (self.directory/'recording.json',self.report/'video_recording.json'):
                        data=json.loads(manifest.read_text(encoding='utf-8'))
                        data.update(conversion_status='FAILED',conversion_error=str(exc))
                        manifest.write_text(json.dumps(data,indent=2),encoding='utf-8')


def retain_32x_video(directory, report=None):
    """Replace a closed recording only after the accelerated copy is validated."""
    from .rollout_recorder import speed_up_mp4
    directory=Path(directory);manifest=directory/'recording.json'
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if data['status'] not in ('COMPLETED','FAILED'):
        raise RuntimeError('recording is not closed; refusing to convert')
    source=directory/('left_wrist_rgb.mp4' if data.get('camera_backend')=='realsense' else 'zed_rgb.mp4');output=source.with_name(source.stem+'_32x.mp4')
    if not source.exists():
        if data.get('playback_speed')==32 and output.exists():return data
        raise RuntimeError('original recording is missing')
    def probe(path):
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,duration:format=duration',
            '-of','json',str(path)],capture_output=True,text=True,check=True)
        info=json.loads(result.stdout);stream=info['streams'][0]
        return float(info['format']['duration']),stream
    original_duration,original_stream=probe(source)
    speed_up_mp4(source,output,speed=32.)
    duration,stream=probe(output)
    # FFmpeg CFR rounding can retain up to three frames for very short clips.
    tolerance=max(.1,3/float(data.get('fps',10))+.05)
    if abs(duration-original_duration/32)>tolerance:
        raise RuntimeError('32x duration verification failed; original retained')
    if (stream['width'],stream['height'])!=(original_stream['width'],original_stream['height']):
        raise RuntimeError('32x dimensions changed; original retained')
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'],
                   capture_output=True,check=True)
    data.update(video=str(output.resolve()),playback_speed=32,original_duration_s=original_duration,
        video_duration_s=duration,original_deleted=False,conversion_status='VERIFIED')
    # Persist the verified replacement before deleting the source.
    paths=[manifest]+([Path(report)/'video_recording.json'] if report else [])
    def save():
        for path in paths:
            tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(path)
    save();source.unlink();data['original_deleted']=True;save()
    return data
