import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_blank_validation_record_is_rejected_without_writing_config(tmp_path):
    record = tmp_path / 'record.json'
    output = tmp_path / 'config.json'
    subprocess.run([
        sys.executable, 'scripts/create_zed_validation_record.py',
        '--output', str(record),
    ], cwd=ROOT, check=True)
    result = subprocess.run([
        sys.executable, 'scripts/apply_zed_validation_record.py',
        '--record', str(record), '--output', str(output),
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert not output.exists()
    payload = json.loads(record.read_text())
    assert payload['execution_allowed'] is False
    assert all(row['operator_result'] is None for row in payload['checks'])
