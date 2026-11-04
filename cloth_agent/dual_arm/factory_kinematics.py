"""Read factory joint origins using UFactory's read-only calibration request.

Protocol source: xArm-Developer/xarm_ros2, gen_kinematics_params.py.
Factory origins use metres/radians, independently of SDK TCP millimetres/degrees.
"""
from pathlib import Path
import socket
import struct
import xml.etree.ElementTree as ET

import numpy as np


def decode_origins(packet, expected_axis):
    if len(packet) != 179 or packet[:4] != bytes.fromhex('00010002'):
        raise ValueError('Invalid factory calibration response')
    if not packet[8] or packet[9] != expected_axis:
        raise ValueError('Factory calibration missing or axis mismatch')
    origins = np.asarray(struct.unpack('<42f', packet[11:])).reshape(7, 6)[:expected_axis]
    if not np.isfinite(origins).all():
        raise ValueError('Nonfinite factory joint origin')
    return origins


def read_origins(address, expected_axis):
    with socket.create_connection((address, 502), timeout=3) as connection:
        connection.settimeout(3)
        connection.sendall(bytes.fromhex('00010002000108'))
        packet = bytearray()
        while len(packet) < 179:
            chunk = connection.recv(179-len(packet))
            if not chunk:
                raise ValueError('Incomplete factory calibration response')
            packet.extend(chunk)
    return decode_origins(packet, expected_axis)


def calibrated_urdf(source, origins):
    source = Path(source).resolve()
    origins = np.asarray(origins, dtype=float)
    if origins.shape not in ((6, 6), (7, 6)) or not np.isfinite(origins).all():
        raise ValueError('Expected six metre/radian origin values per joint')
    tree = ET.parse(source)
    for index, values in enumerate(origins, start=1):
        joint = tree.find(f"joint[@name='joint{index}']")
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError(f'Missing revolute joint{index}')
        origin = joint.find('origin')
        if origin is None:
            raise ValueError(f'Missing origin for joint{index}')
        origin.set('xyz', ' '.join(format(value, '.17g') for value in values[:3]))
        origin.set('rpy', ' '.join(format(value, '.17g') for value in values[3:]))
    for mesh in tree.findall('.//mesh'):
        filename = mesh.get('filename')
        if not filename or '://' in filename:
            raise ValueError('Expected filesystem mesh paths')
        mesh.set('filename', str((source.parent / filename).resolve()))
    return ET.tostring(tree.getroot(), encoding='unicode') + '\n'
