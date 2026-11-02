"""Host-owned durable offline session journal and version records."""
from __future__ import annotations
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

TASK = 'Flatten a randomly crumpled T-shirt on the table, making its main body, sleeves and hem visible and accessible.'
INSTRUCTIONS = '''Use observations and basic robot tools to discover a solution.
Explore: propose actions from observations, inspect outcomes and record evidence.
Compile: write code/<project>/controller.py with main(ctx), reading new observations
and computing actions each run. Evaluate: execute the program and inspect results.
Improve: revise from observed failures; preserve versions and test evidence.
Keep notes in memory/ and skills with evidence in skill-library/.
Mock is an interface fixture. Replay is historical evidence; neither predicts a new
action's physical outcome. UNKNOWN safety never permits execution. No arm tool exists.
Only task-independent tool documentation is supplied. No initial garment skills.
'''


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


class Session:
    def __init__(self, root, config, *, resume=False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = self.root / 'state.json'
        if path.exists():
            if not resume:
                self.close()
                raise ValueError('session exists; use --resume')
            self.state = json.loads(path.read_text())
            if self.state['config'] != config:
                self.close()
                raise ValueError('resume configuration differs; reuse exact mode, seed and budgets')
        else:
            if resume:
                self.close()
                raise ValueError('cannot resume a missing session')
            self.state = dict(config=config, created_unix_s=time.time(), episodes=[],
                              llm_calls=0, llm_cost_usd=0.0, input_tokens=0, output_tokens=0,
                              elapsed_seconds=0.0, physical_actions=0, physical_seconds=None,
                              task_success=None, status='created')
            self.save()
        ws = self.root / 'agent-workspace'
        for part in ('code', 'memory', 'skills', 'skill-library'):
            (ws / part).mkdir(parents=True, exist_ok=True)
        for name, content in [('TASK.md', TASK+'\n'), ('AGENTS.md', INSTRUCTIONS)]:
            if not (ws / name).exists():
                (ws / name).write_text(content)
        # Incomplete episodes are never replayed as successful completed operations.
        for row in self.state['episodes']:
            if row['status'] == 'running':
                row['status'] = 'interrupted'
        self.save()

    def save(self):
        write_json(self.root / 'state.json', self.state)

    def audit(self, event, **data):
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event, unix_s=time.time(), **data), allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())

    def episode(self):
        n = len(self.state['episodes'])
        folder = self.root / 'episodes' / f'{n:06d}'
        folder.mkdir(parents=True, exist_ok=False)
        row = dict(id=n, status='running', task_success=None, physical_actions=0)
        self.state['episodes'].append(row)
        self.save()
        return folder, row

    def version(self, source, episode, author='external_program'):
        # Syntax validation does NOT execute generated code or imply sandbox security.
        source = str(source)
        digest = hashlib.sha256(source.encode()).hexdigest()
        folder = self.root / 'programs' / digest
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'controller.py').write_text(source)
        result = dict(sha256=digest, author=author, episode=episode,
                      syntax_valid=False, execution='NOT_RUN', physical_verified=False)
        try:
            tree = ast.parse(source)
            if not any(isinstance(x, ast.FunctionDef) and x.name == 'main' for x in tree.body):
                raise ValueError('controller.py must define main(ctx)')
            result['syntax_valid'] = True
        except (SyntaxError, ValueError) as exc:
            result['error'] = str(exc)
        write_json(folder / f'test-{episode:06d}.json', result)
        self.audit('program_version', **result)
        return result

    def finish(self, status):
        self.state['status'] = status
        self.save()
        write_json(self.root / 'results.json', self.state)
        done = sum(x['status'] == 'complete' for x in self.state['episodes'])
        (self.root / 'summary.md').write_text(
            '# RAX-Cloth offline setting test\n\n'
            f'Mode: {self.state["config"]["mode"]}; status: {status}; completed interface episodes: {done}.\n\n'
            'Physical actions: 0. Task success: NOT_EVALUATED. Cloth deformation feedback: unavailable.\n'
            f'LLM calls: {self.state["llm_calls"]}; cost: ${self.state["llm_cost_usd"]:.4f}.\n'
            'No agent exploration was performed. Fixture code is labeled test_fixture.\n'
            'Exploration cost, compiled-program inference, and physical execution are separate; '
            'unmeasured quantities are null, never fabricated zeros.\n')

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
