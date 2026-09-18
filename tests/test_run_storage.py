import json
from pathlib import Path
import pytest
from cloth_agent.run_storage import find_run, iter_runs, new_run_path, runs_root


def configure(root, target, **extra):
    (root / 'config').mkdir(parents=True)
    (root / 'config/run_storage.json').write_text(json.dumps({'runs_root': str(target), **extra}))


def test_date_layout_resume_and_legacy(tmp_path):
    root = tmp_path / 'project'
    configure(root, tmp_path / 'ssd/runs')
    run = new_run_path(root, 'fold_test')
    assert run.parent.parent == tmp_path / 'ssd/runs'
    assert len(run.parent.name) == 10
    run.mkdir(parents=True)
    (run / 'run_metadata.json').write_text('{}')
    legacy = root / 'runs/old'
    legacy.mkdir(parents=True)
    (legacy / 'run_metadata.json').write_text('{}')
    assert find_run(root, 'fold_test') == run
    assert set(iter_runs(root)) == {run, legacy}
    with pytest.raises(FileExistsError):
        new_run_path(root, 'fold_test')


def test_unmounted_disk_fails_without_fallback(tmp_path):
    root = tmp_path / 'project'
    configure(root, tmp_path / 'ssd/runs', required_mount=str(tmp_path / 'not_mounted'))
    with pytest.raises(RuntimeError, match='not mounted'):
        new_run_path(root, 'test')
    assert not (root / 'runs').exists()


def test_reject_traversal_ambiguity_and_symlink(tmp_path):
    root = tmp_path / 'project'
    target = tmp_path / 'ssd/runs'
    configure(root, target)
    for name in ('../escape', '/', '.', ''):
        with pytest.raises(ValueError):
            find_run(root, name)
    for day in ('2026-09-16', '2026-09-17'):
        (target / day / 'duplicate').mkdir(parents=True)
    with pytest.raises(ValueError, match='Ambiguous'):
        find_run(root, 'duplicate')
    (target / 'escape').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(PermissionError):
        find_run(root, 'escape')


def test_literal_id(tmp_path):
    run = tmp_path / 'runs/a[1]'
    run.mkdir(parents=True)
    assert find_run(tmp_path, 'a[1]') == run


def test_session_external_storage_and_read_isolation(tmp_path):
    from cloth_agent.config import RobotConfig, WorkspaceBounds, ExperimentConfig
    from cloth_agent.session import AgentSession
    from cloth_agent.free_exploration import _load_or_create_session
    root = tmp_path / 'project'
    configure(root, tmp_path / 'ssd/runs')
    robot = RobotConfig(robot_ip='test', boundaries=WorkspaceBounds(
        x_min=0, x_max=600, y_min=-300, y_max=300, z_min=0, z_max=500),
        init_joints_deg=(0,)*6, init_pose_mm_deg=(300,0,200,180,0,0),
        orientation_roll_deg=180, orientation_pitch_deg=0)
    session = AgentSession.create(root, 'storage test', robot, ExperimentConfig(), run_id='test')
    assert session.run_dir.parent.parent == tmp_path / 'ssd/runs'
    assert 'test' in session.inspect_file(session.run_dir / 'run_metadata.json')
    (root / 'xarm_boundaries.json').write_text('{}')
    (root / 'data/robot').mkdir(parents=True)
    (root / 'data/robot/xarm_init_pose.json').write_text(json.dumps({
        'joint_angles_deg': [0]*6, 'tcp_pose_mm_deg': [300,0,200,180,0,0]}))
    reopened = _load_or_create_session(root, None, 'test', None)
    assert reopened.run_dir == session.run_dir
    other = AgentSession.create(root, 'other', robot, ExperimentConfig(), run_id='other')
    with pytest.raises(PermissionError):
        session.inspect_file(other.run_dir / 'run_metadata.json')
    assert not (root / 'runs').exists()
