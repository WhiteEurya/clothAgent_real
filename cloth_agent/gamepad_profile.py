"""Device-specific mappings, activated only after guided verification passes."""
import json
from pathlib import Path

PROFILE_ROOT = Path(__file__).resolve().parents[1] / 'config/gamepads'
BUTTONS = ('A', 'B', 'X', 'Y', 'LB', 'START')


def identity(device):
    return {key: getattr(device.info, key) for key in ('bustype', 'vendor', 'product', 'version')}


def profile_path(device):
    return PROFILE_ROOT / ('_'.join(f'{value:04x}' for value in identity(device).values()) + '.json')


def load_profile(device):
    if not hasattr(device, 'info'):
        return None
    path = profile_path(device)
    return json.loads(path.read_text()) if path.exists() else None


def validate_profile(profile, device):
    if profile.get('version') != 1 or profile.get('device') != identity(device):
        raise ValueError('gamepad profile identity/version mismatch')
    axes = profile.get('axes', [])
    buttons = profile.get('buttons', {})
    if len(axes) != 3 or any(not isinstance(a, dict) or type(a.get('code')) is not int
                            or type(a.get('sign')) is not int or a['sign'] not in (-1, 1) for a in axes):
        raise ValueError('invalid gamepad axes/signs')
    if len({a['code'] for a in axes}) != 3:
        raise ValueError('stick axes must be distinct')
    if set(buttons) != set(BUTTONS) or any(type(v) is not int for v in buttons.values()) or len(set(buttons.values())) != 6:
        raise ValueError('buttons must have six distinct codes')
    caps = device.capabilities(absinfo=False)
    if not {a['code'] for a in axes} <= set(caps.get(3, [])) or not set(buttons.values()) <= set(caps.get(1, [])):
        raise ValueError('saved gamepad profile does not match available controls')


def save_profile(profile, device):
    validate_profile(profile, device)
    path = profile_path(device)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + '\n')
    pending.replace(path)
    return path
