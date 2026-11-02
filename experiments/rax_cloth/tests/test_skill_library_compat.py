import subprocess

from rax_cloth.baseline import configure, prepare


def commit(repo):
    for args in [('add', '-A'), ('-c', 'user.name=test', '-c', 'user.email=test@local',
                                'commit', '-qm', 'Test draft persistence')]:
        subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True)


def test_original_merge_and_cross_session_checkout(tmp_path):
    _, cfg = configure(tmp_path/'first', 'mock')
    ws = prepare(cfg)
    from services import skill_library as lib
    repo = ws/'skill-library'
    skill = repo/'skills'/'interface-fixture'
    (skill/'code').mkdir(parents=True)
    (skill/'evidence'/'images').mkdir(parents=True)
    # A test artifact only, never an initial garment skill or physical evidence.
    import base64
    (skill/'evidence'/'images'/'fixture.png').write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII='))
    (skill/'code'/'run.py').write_text('def main(ctx):\n    raise RuntimeError("test fixture only")\n')
    (skill/'SKILL.md').write_text('---\nname: interface-fixture\ndescription: Test fixture only\n'
        'status: draft\nrobot: cloth\nentry: code/run.py\n---\n'+
        '\n'.join('## '+s+'\nTest fixture only.\n' for s in lib.SECTIONS)+
        '\n### Perception checkpoints\nNone\n')
    with (repo/'INDEX.md').open('a') as f:
        f.write('| [interface-fixture](skills/interface-fixture/SKILL.md) | draft | fixture | test |\n')
    commit(repo)
    result = lib.merge(cfg, ws, 'test fixture')
    assert result['merged'] is True
    _, second = configure(tmp_path/'second', 'mock')
    next_ws = prepare(second)
    saved = next_ws/'skill-library'/'skills'/'interface-fixture'
    assert (saved/'code'/'run.py').read_bytes() == (skill/'code'/'run.py').read_bytes()
    assert 'status: draft' in (saved/'SKILL.md').read_text()
    # Compatibility must not bypass the original evidence validator.
    path = saved/'SKILL.md'
    path.write_text(path.read_text().replace('status: draft', 'status: verified'))
    commit(next_ws/'skill-library')
    refused = lib.merge(second, next_ws, 'invalid promotion')
    assert refused['merged'] is False
    assert any('verified needs' in p for p in refused['problems'])
