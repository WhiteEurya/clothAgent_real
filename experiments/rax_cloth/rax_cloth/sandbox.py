"""Fail-closed program isolation probe. Never use subprocess execution as a sandbox."""
import shutil
import subprocess


def probe():
    bwrap = shutil.which('bwrap')
    if not bwrap:
        return {'available': False, 'reason': 'bubblewrap not installed'}
    result = subprocess.run([bwrap, '--unshare-all', '--die-with-parent', '--new-session',
                             '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
                             '--ro-bind', '/lib64', '/lib64', '/usr/bin/true'],
                            capture_output=True, text=True, timeout=10)
    return dict(available=result.returncode == 0, reason=result.stderr.strip(),
                returncode=result.returncode)


def require_agent_ready():
    result = probe()
    if not result['available']:
        raise PermissionError('Agent/program execution disabled: isolation unavailable: '+result['reason'])
    # A namespace probe alone does not verify CLI file tools, relay or billing.
    raise PermissionError('Full pinned upstream launcher, protected mounts and model budget relay must be validated first')
