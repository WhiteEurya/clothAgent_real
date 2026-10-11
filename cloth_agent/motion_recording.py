"""Record RGB from the already-open observation camera during motion only."""
import json
import threading
import time
from pathlib import Path

import cv2


class MotionRecorder:
    def __init__(self,camera,folder,*,fps=30,camera_label='A'):
        self.camera=camera;self.folder=Path(folder);self.fps=float(fps)
        self.path=self.folder/'execution.mp4';self.label=camera_label
        self.stop_event=threading.Event();self.thread=None;self.writer=None
        self.frames=0;self.error=None;self.started=None;self.timestamps=[]

    def start(self):
        self.started=time.time();self.clock=time.monotonic()
        try:
            rgb,_=self.camera.read()
            h,w=rgb.shape[:2]
            self.writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(w,h))
            if not self.writer.isOpened():raise RuntimeError('cannot open MP4 video encoder')
            self._write(rgb)
            self.thread=threading.Thread(target=self._run,name='motion-video',daemon=True)
            self.thread.start()
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            self.stop()
            raise
        print(f'开始录像：{self.path}',flush=True)

    def _write(self,rgb):
        self.writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        self.timestamps.append(time.monotonic()-self.clock);self.frames+=1

    def _run(self):
        try:
            while not self.stop_event.is_set():
                rgb,_=self.camera.read()
                if not self.stop_event.is_set():self._write(rgb)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
            print(f'录像中断（运动结束后保存已录内容）：{self.error}',flush=True)
        finally:
            self.writer.release()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            # RealSense wait_for_frames has a bounded default timeout. Never
            # let the next observation compete with this reader.
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                self.camera.stop()
                self.thread.join(timeout=6)
                if self.thread.is_alive():raise RuntimeError('video camera reader did not stop')
        elif self.writer is not None:self.writer.release()
        report=dict(path=str(self.path),camera=self.label,frames=self.frames,fps=self.fps,
                    started_unix_s=self.started,elapsed_s=time.monotonic()-self.clock,
                    status='FAILED' if self.error else 'RECORDED',error=self.error,
                    frame_times_s=self.timestamps)
        (self.folder/'recording.json').write_text(json.dumps(report,indent=2)+'\n')
        return {k:v for k,v in report.items() if k!='frame_times_s'}
