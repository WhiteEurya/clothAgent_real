"""Load the reviewed upstream registry, or register with a complete pinned checkout."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def register_checkout():
    """Load the verified original registry, never the Rainbow hardware backend."""
    manifest = json.loads((ROOT / 'upstream.json').read_text())
    checkout = ROOT.parent / 'vendor' / 'rax'
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if not manifest['full_checkout'] or actual != manifest['commit']:
        raise ValueError('full upstream checkout does not match pinned commit')
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=normal'], text=True)
    if dirty.strip():
        raise ValueError('upstream checkout must remain unmodified: '+dirty[:500])
    sys.path.insert(0, str(checkout)) if str(checkout) not in sys.path else None
    from services import backends
    if Path(backends.__file__).resolve() != checkout.resolve() / 'services/backends.py':
        raise ValueError('services namespace shadowed by another package')
    backends.BACKENDS['cloth'] = 'rax_cloth.backend:ClothBackend'
    return checkout, backends


def make_backend(root, options):
    manifest = json.loads((ROOT / 'upstream.json').read_text())
    if manifest.get('full_checkout'):
        _, registry = register_checkout()
        return registry.make_backend(SimpleNamespace(root=str(root), robot='cloth', backend_options=options))
    path = ROOT / 'upstream_reference' / 'backends.py'
    manifest = json.loads((ROOT / 'upstream.json').read_text())
    expected = manifest['reference_files']['backends.py']['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('upstream registry snapshot hash mismatch')
    spec = importlib.util.spec_from_file_location('_rax_reference_backends', path)
    registry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry)
    # Session-local registration; no production or upstream source edits.
    registry.BACKENDS['cloth'] = 'rax_cloth.backend:ClothBackend'
    return registry.make_backend(SimpleNamespace(root=str(root), robot='cloth', backend_options=options))
