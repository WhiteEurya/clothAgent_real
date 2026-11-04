"""ZED rectified LEFT RGB + aligned optical-Z depth in meters; no robot access."""
from __future__ import annotations

import numpy as np

from .perception import PerceptionError


def load_zed_sdk():
    """Import only; expose loader errors without opening a camera or robot."""
    import sys
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise PerceptionError(
            f'ZED SDK import failed in {sys.executable}: {exc}. '
            'Check SDK library paths and directory access; /usr/local/zed may require the zed group.'
        ) from exc
    return sl


class ZEDRGBD:
    def __init__(self, spec, width, height, fps):
        self.spec, self.width, self.height, self.fps = spec, width, height, fps
        self.started = False
        self.intrinsics = None
        self.camera = None
        self.timestamp_ns = None
        self.confidence = None

    def start(self):
        if self.started:
            return
        resolutions = {(1280, 720): 'HD720', (1920, 1080): 'HD1080',
                       (2208, 1242): 'HD2K', (672, 376): 'VGA'}
        if (self.width, self.height) not in resolutions:
            raise PerceptionError('Unsupported ZED resolution')
        if self.spec.depth_mode not in ('ULTRA', 'NEURAL', 'NEURAL_LIGHT'):
            raise PerceptionError('ZED requires an explicit supported depth mode')
        threshold = self.spec.confidence_threshold
        if type(threshold) is not int or not 0 <= threshold <= 100:
            raise PerceptionError('ZED confidence_threshold must be an integer in [0, 100]')
        if any(v is not None for v in (self.spec.color_exposure, self.spec.color_white_balance,
                                      self.spec.depth_exposure_us, self.spec.depth_gain)):
            raise PerceptionError('RealSense exposure controls cannot be applied to ZED')
        sl = load_zed_sdk()
        self.sl = sl
        params = sl.InitParameters()
        params.set_from_serial_number(int(self.spec.serial))
        params.camera_resolution = getattr(sl.RESOLUTION, resolutions[self.width, self.height])
        params.camera_fps = self.fps
        params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        params.coordinate_units = sl.UNIT.METER
        params.depth_mode = getattr(sl.DEPTH_MODE, self.spec.depth_mode)
        params.depth_minimum_distance, params.depth_maximum_distance = .15, 2.
        self.camera = sl.Camera()
        try:
            self._check(self.camera.open(params), 'open')
            info = self.camera.get_camera_information()
            config = info.camera_configuration
            if str(info.serial_number) != self.spec.serial:
                raise PerceptionError('ZED serial does not match calibration')
            if (config.resolution.width, config.resolution.height) != (self.width, self.height):
                raise PerceptionError('ZED resolution does not match calibration')
            left = config.calibration_parameters.left_cam
            self.intrinsics = np.array([[left.fx, 0., left.cx], [0., left.fy, left.cy], [0., 0., 1.]])
            self.runtime = sl.RuntimeParameters()
            self.runtime.confidence_threshold = threshold
            self.runtime.texture_confidence_threshold = 100
            self.runtime.enable_fill_mode = False
            self.image, self.depth, self.quality = sl.Mat(), sl.Mat(), sl.Mat()
            self.started = True
        except BaseException:
            self.stop()
            raise

    def _check(self, status, operation):
        if status != self.sl.ERROR_CODE.SUCCESS:
            raise PerceptionError(f'ZED {operation} failed: {status}')

    def read(self):
        if not self.started:
            raise PerceptionError('ZED has not been started')
        sl = self.sl
        self._check(self.camera.grab(self.runtime), 'grab')
        self._check(self.camera.retrieve_image(self.image, sl.VIEW.LEFT), 'left image')
        self._check(self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH), 'depth')
        self._check(self.camera.retrieve_measure(self.quality, sl.MEASURE.CONFIDENCE), 'confidence')
        # ZED SDK image memory is BGRA. The perception API uses RGB.
        rgb = self.image.get_data()[..., :3][..., ::-1].copy()
        depth = np.array(self.depth.get_data(), dtype=np.float32, copy=True)
        self.confidence = self.quality.get_data().copy()
        if rgb.shape != (self.height, self.width, 3) or depth.shape != (self.height, self.width) or self.confidence.shape != depth.shape:
            raise PerceptionError('ZED RGB/depth/confidence dimensions differ')
        invalid = (~np.isfinite(depth) | (depth < .15) | (depth > 2.)
                   | ~np.isfinite(self.confidence) | (self.confidence > self.spec.confidence_threshold))
        depth[invalid] = np.nan
        self.timestamp_ns = int(self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        return rgb, depth

    def stop(self):
        try:
            if self.camera is not None:
                self.camera.close()
        finally:
            self.camera = None
            self.started = False
