"""Load R2 configuration without executing shell code."""
import os
import shlex
from pathlib import Path


def load_r2_env():
    path = Path.home() / '.config/clothagent/r2.env'
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith('export '):
            line = line[7:].lstrip()
        name, sep, value = line.partition('=')
        name = name.strip()
        if not sep or name not in {'R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET'}:
            continue
        parts = shlex.split(value, comments=True)
        if len(parts) == 1:
            os.environ.setdefault(name, parts[0])
