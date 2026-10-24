import json
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from cloth_agent.experiments.trajectory_memory_agent_probe import (
    AccessMonitor, access_summary, isolated_command, prepare, read, save, validate_plan,
)


def test_existing_query_embargoes_entire_session_and_future(tmp_path):
    data = tmp_path/"data"
    for session, step, color in [("20261019s1", "traj_0000", "red"),
                                  ("20261019s1", "traj_0001", "blue"),
                                  ("20261019s2", "traj_0000", "green")]:
        folder = data/session/step
        save(folder/"traj.json", dict(status="COMPLETE", operator_accepted=True,
                                      collection_mode="automatic_pick_place"))
        for sub, shade in [("observation", color), ("after", "white")]:
            (folder/sub).mkdir()
            Image.new("RGB", (32, 24), shade).save(folder/sub/"rgb.png")
        (folder/"events.jsonl").write_text("")
        (folder/"states.jsonl").write_text("")
        save(folder/"automatic_pick_place/result.json", {"status": "EXECUTED"})
        (folder/"automatic_pick_place/motion.py").write_text("DO_NOT_EXECUTE_THIS_SOURCE")
        save(folder/"points.json", [])
    queries = []
    for i, color in enumerate(["red", "yellow", "black"]):
        path = tmp_path/f"q{i}.png"
        Image.new("RGB", (32, 24), color).save(path)
        queries.append(dict(id=f"q{i}", rgb=str(path), fresh_capture=i != 0,
                            source_session="20261019s1" if i == 0 else None))
    spec = tmp_path/"spec.json"
    save(spec, dict(data_root=str(data), scope_dates=["20261019"], queries=queries))
    output = tmp_path/"probe"
    prepare(output, spec)
    manifest = read(output/"manifest.json")
    assert manifest["memory_steps"] == 1
    assert manifest["embargoed_sessions"] == ["20261019s1"]
    assert not (output/"memory/20261019s1").exists()
    assert (output/"memory/20261019s2/traj_0000/automatic_pick_place_motion.py.txt").exists()


def test_access_audit_counts_reads_not_catalog_entries(tmp_path):
    before, after = tmp_path/"before.png", tmp_path/"after.png"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    prefix = "/memory/s1/traj_0000"
    monitor = AccessMonitor([(before, prefix+"/before.png"), (after, prefix+"/after.png")]).start()
    before.read_bytes()
    access = monitor.stop()
    assert prefix+"/before.png" in access["files"]
    assert prefix+"/after.png" not in access["files"]
    records = [dict(trajectory_id="s1", step_id="traj_0000", directory=prefix)]
    plan = dict(supporting_experiences=[dict(trajectory_id="s1", step_id="traj_0000", use="test")])
    result = access_summary(access, records, plan)
    assert result["historical_source_images_read"] == 1
    assert result["memory_step_files_read"] == 1
    assert not result["cited_experience_read_checks"][0]["after_read"]


def test_plan_citations_and_pixel_bounds():
    plan = dict(decision="ACTION", grasp_px=[10, 20], target_px=[30, 40],
                motion_type="lift_pull_place", supporting_experiences=[], short_rationale="test")
    validate_plan(plan, [1280, 720], "no_memory", [])
    plan["target_px"] = [1280, 40]
    with pytest.raises(ValueError, match="outside"):
        validate_plan(plan, [1280, 720], "no_memory", [])
    plan["decision"] = "ABSTAIN"
    with pytest.raises(ValueError, match="ABSTAIN"):
        validate_plan(plan, [1280, 720], "no_memory", [])
    plan["grasp_px"] = plan["target_px"] = None
    plan["supporting_experiences"] = [dict(trajectory_id="fake", step_id="fake", use="invented")]
    with pytest.raises(ValueError, match="fabricated"):
        validate_plan(plan, [1280, 720], "no_memory", [])


def test_namespace_hides_host_data_and_baseline_memory(tmp_path):
    work = tmp_path/"work"
    work.mkdir()
    (work/"query.txt").write_text("query only")
    code = (
        "from pathlib import Path; import PIL,numpy,json; "
        "assert not Path('/home/sja/clothAgent_real').exists(); "
        "assert not Path('/mnt/newssd').exists(); "
        "assert not Path('/memory').exists(); "
        "assert not Path('/dev/bus/usb').exists(); "
        "assert Path('/opt/codex-package/codex-resources/bwrap').exists(); "
        "assert Path('/work/query.txt').read_text() == 'query only'; "
        "Path('/work/helper.py').write_text('# temporary helper'); "
        "print('ISOLATION_OK')"
    )
    result = subprocess.run(isolated_command(work)+["/usr/bin/python3", "-c", code],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "ISOLATION_OK" in result.stdout
