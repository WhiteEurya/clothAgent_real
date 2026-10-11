"""Camera B calibration shared by single-arm and dual-arm entrypoints."""
import json
import hashlib
from copy import deepcopy
import numpy as np
import yaml
from .dual_arm.config import DualConfig
from .dual_arm.geometry import DualArmError


def with_arm7_camera(config):
    raw=deepcopy(config.raw)
    pose_file=config.root/'data/robot/xarm7_perception_pose.json'
    extrinsic_file=config.root/'config/calibration/dual_arm_working_20261008/camB_extrinsics.yaml'
    pose=json.loads(pose_file.read_text())
    extrinsic=yaml.safe_load(extrinsic_file.read_text())
    matrix=np.asarray(extrinsic['X_CammountCam'],dtype=float).copy()
    units=extrinsic.get('translation_units')
    if units=='m':matrix[:3,3]*=1000
    elif units!='mm':raise DualArmError('Camera B extrinsic translation units missing/unsupported')
    camera={k:raw['camera'][k] for k in ('width','height','fps')}
    camera.update(label='B',serial=str(pose['camera_serial']),mount_arm='right',mount_link='link_eef',
        mount_from_camera_mm=matrix.tolist(),perception_pose_file=str(pose_file),
        extrinsics_source=str(extrinsic_file),depth_exposure_us=4000.,depth_gain=16.)
    raw['camera']=camera # Do not inherit arm6's empirical 1-degree correction.
    suffix=hashlib.sha256(json.dumps(camera,sort_keys=True).encode()).hexdigest()[:12]
    raw['calibration_id']+=f'_arm7_cameraB_{suffix}'
    return DualConfig.parse(raw,config.root,homing_only=True)
