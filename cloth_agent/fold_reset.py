"""Durable human reset requests; confirmation never sends robot commands."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


RESET_INSTRUCTION = (
    "You may proactively request a HUMAN garment reset when current RGB and recent "
    "attempts show that the garment needs manual repositioning to continue: for example "
    "it is tangled, displaced out of reach/view, or repeated attempts cannot make useful "
    "progress. Set trajectory_decision=REQUEST_RESET, status=BLOCKED, current_step=BLOCKED. "
    "Give concrete visual evidence and explain in reason what the human should restore. "
    "A single failed grasp, border contact alone, or a network/model error is not sufficient. "
    "This is a request for human intervention, never a robot motion plan. The host pauses "
    "until explicit human confirmation and then observes the garment afresh."
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class FoldReset:
    def __init__(self, workspace):
        self.state_path = Path(workspace) / 'fold_reset' / 'state.json'

    def state(self):
        return _read(self.state_path) if self.state_path.exists() else {}

    def pending(self):
        state = self.state()
        return state if state.get('status') == 'WAITING_FOR_RESET' else None

    def current_history(self, history):
        """Keep archived experience, but discard pre-reset physical-state claims."""
        boundary = self.state().get('confirmed_at')
        if not boundary:
            return list(history)
        cutoff = datetime.fromisoformat(boundary)
        return [row for row in history if row.get('created_at')
                and datetime.fromisoformat(row['created_at']) > cutoff]

    def request(self, directory, decision, *, stage):
        if self.pending():
            raise RuntimeError('A garment reset is already awaiting confirmation')
        if decision.get('trajectory_decision') != 'REQUEST_RESET' or decision.get('fallback'):
            raise ValueError('Only an explicit Claude decision can request a garment reset')
        reason, evidence = decision.get('reason'), decision.get('evidence')
        if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, list) or not evidence:
            raise ValueError('Reset requires a reason and evidence')
        path = (Path(directory) / 'reset_request.json').resolve()
        request = dict(request_id=uuid.uuid4().hex, status='WAITING_FOR_RESET',
                       requested_at=_now(), stage=stage, reason=reason, evidence=evidence,
                       request_path=str(path), state_path=str(self.state_path.resolve()))
        request['confirmation_command'] = shlex.join([
            sys.executable, '-m', 'cloth_agent.fold_reset', '--request', str(path),
            '--request-id', request['request_id'], '--confirm',
        ])
        # Persist the gate first: an interrupted write must not permit a restart to move.
        _write(self.state_path, request)
        _write(path, request)
        return request

    def wait(self, request, *, poll_s=0.5):
        # Recover the human-facing file if the process died after writing the gate.
        _write(request['request_path'], request)
        confirmation = Path(request['request_path']).with_name('reset_confirmation.json')
        while True:
            if confirmation.is_file():
                receipt = _read(confirmation)
                if (receipt.get('request_id') == request['request_id']
                        and receipt.get('confirmed') is True):
                    completed = {**request, 'status': 'CONFIRMED', 'confirmed_at': _now()}
                    _write(self.state_path, completed)
                    _write(request['request_path'], completed)
                    return completed
            time.sleep(poll_s)


def confirm(request_path, request_id):
    request = _read(request_path)
    current = _read(request['state_path'])
    if (request.get('request_id') != request_id
            or current.get('request_id') != request_id
            or current.get('status') != 'WAITING_FOR_RESET'):
        raise ValueError('This reset request is stale or is not awaiting confirmation')
    receipt = {'request_id': request_id, 'confirmed': True, 'confirmed_at': _now()}
    _write(Path(request_path).with_name('reset_confirmation.json'), receipt)


def main():
    parser = argparse.ArgumentParser(description='Confirm the requested manual garment reset is complete.')
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--confirm', action='store_true', required=True,
                        help='Confirm you have repositioned the garment and cleared the workspace.')
    args = parser.parse_args()
    confirm(args.request, args.request_id)
    print('已确认人工 reset；等待中的流程将重新采图并判断衣物状态。')


if __name__ == '__main__':
    main()
