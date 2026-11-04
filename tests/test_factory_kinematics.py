import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cloth_agent.dual_arm.factory_kinematics import calibrated_urdf, decode_origins


def packet(origins, valid=1, axis=7):
    return bytes.fromhex('0001000200ad0800') + bytes([valid, axis, 7]) + struct.pack('<42f', *origins)


def test_decode_preserves_metre_radian_origins():
    values = np.arange(42, dtype=float) / 100
    np.testing.assert_allclose(decode_origins(packet(values), 7), values.reshape(7, 6))


@pytest.mark.parametrize('response', [b'', packet([0.]*42, valid=0),
    packet([0.]*42, axis=6), packet([float('nan')]*42)])
def test_invalid_calibration_rejected(response):
    with pytest.raises(ValueError):
        decode_origins(response, 7)


def test_urdf_changes_only_joint_origins_and_resolves_meshes(tmp_path):
    source = tmp_path / 'source.urdf'
    joints = ''.join(f'<joint name="joint{index}" type="revolute"><origin xyz="0 0 0" '
        f'rpy="0 0 0"/><limit lower="-2" upper="2"/></joint>' for index in range(1, 8))
    source.write_text('<robot name="test">' + joints +
        '<joint name="tcp" type="fixed"><origin xyz="0 0 .172" rpy="0 0 0"/></joint>'
        '<link name="tool"><visual><geometry><mesh filename="meshes/tool.stl"/>'
        '</geometry></visual></link></robot>')
    origins = np.zeros((7, 6))
    origins[2] = [.001, -.293, .002, 1.57, .003, -.004]
    original = source.read_bytes()
    result = ET.fromstring(calibrated_urdf(source, origins))
    assert source.read_bytes() == original
    np.testing.assert_allclose(np.fromstring(result.find("joint[@name='joint3']/origin").get('xyz'), sep=' '), origins[2, :3])
    assert result.find("joint[@name='joint3']/limit").get('lower') == '-2'
    assert result.find("joint[@name='tcp']/origin").get('xyz') == '0 0 .172'
    assert result.find('.//mesh').get('filename') == str(tmp_path / 'meshes/tool.stl')


@pytest.mark.parametrize('arm,joints,feedback', [
    ('left', [2.837287,-43.951421,1.822865,68.1688,3.40738,111.084542,6.110824],
     [200.968857,34.991394,447.24649,-178.022227,-.817611,-.706858]),
    ('right', [-3.422793,-40.851719,-2.57894,71.689625,-2.013603,112.52364,-4.741684],
     [207.821487,-39.306442,454.290466,178.02767,.169366,-.590548]),
])
def test_deployed_model_matches_stationary_controller_snapshot(arm,joints,feedback):
    from pathlib import Path
    from cloth_agent.dual_arm.config import DualConfig
    from cloth_agent.dual_arm.geometry import pose_error
    from cloth_agent.dual_arm.kinematics import ArmModel
    root=Path(__file__).resolve().parents[1]
    config=DualConfig.load(root/'config/dual_arm.zed_dual_xarm7_20261103.json',root=root)
    distance,angle=pose_error(ArmModel(config.arms[arm]).forward(joints),feedback)
    assert distance < .1
    assert angle < .02
