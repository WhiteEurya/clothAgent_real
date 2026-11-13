"""Original left-wrist RGB video sharing its native RGB-D stream with observation."""
from types import SimpleNamespace
import numpy as np
from .unfold_video import UnfoldVideo


class NativeRealSense:
    def __init__(self,serial):self.serial=serial;self.pipe=None;self.native_frames=None
    def start(self):
        import pyrealsense2 as rs
        self.pipe=rs.pipeline();cfg=rs.config();cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color,1280,720,rs.format.rgb8,30);cfg.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30)
        profile=self.pipe.start(cfg);i=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics=np.array([[i.fx,0,i.ppx],[0,i.fy,i.ppy],[0,0,1.]])
        for _ in range(60):self.pipe.wait_for_frames(5000)
    def read(self):
        frames=self.pipe.wait_for_frames(5000);frames.keep();self.native_frames=frames
        c,d=frames.get_color_frame(),frames.get_depth_frame();self.timestamp_ns=int(c.get_timestamp()*1e6)
        return np.asanyarray(c.get_data()).copy(),np.asanyarray(d.get_data()).copy()
    def stop(self):
        if self.pipe is not None:self.pipe.stop();self.pipe=None


class WristVideo(UnfoldVideo):
    def __init__(self,serial,directory,report):
        spec=SimpleNamespace(serial=serial,label='left_wrist',backend='realsense')
        cfg=SimpleNamespace(cameras=[spec],active_camera_labels=['left_wrist'],width=1280,height=720,fps=30)
        super().__init__(cfg,directory,report,camera=NativeRealSense(serial))
    def read_native_frames(self):
        # Wait for a post-request frame, rather than recycling a moving-arm frame.
        with self.condition:
            start=self.seq
            ok=self.condition.wait_for(lambda:self.seq>start or self.error or self.stop_event.is_set(),timeout=10)
            if not ok or self.error or self.stop_event.is_set():raise RuntimeError(self.error or 'wrist RGB-D frame timed out')
            return self.latest_native_frames
