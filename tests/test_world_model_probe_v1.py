import numpy as np
import pytest

from cloth_agent.experiments.world_model_probe_v1 import (
    CONTINUOUS, assign_split, evaluate, executed_geometry, explicit_failure,
    group_sessions, make_model, predict, shuffled_donors, standardize_actions,
)


def test_connected_sessions_and_duplicate_images_never_cross_splits():
    rows = [dict(session=f"s{i}", before_pixel_sha256=f"b{i}", after_pixel_sha256=f"a{i}")
            for i in range(12)]
    rows[2]["before_pixel_sha256"] = rows[1]["after_pixel_sha256"]
    group_sessions(rows, [("s0", "s1")])
    splits = assign_split(rows, 20261020)
    assert rows[0]["group"] == rows[1]["group"] == rows[2]["group"]
    assert len({r["split"] for r in rows[:3]}) == 1
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    assert set(splits["train"]).isdisjoint(splits["validation"])


def test_action_scaler_sees_only_fit_subset_and_never_imputes_missing():
    rows = [dict(id=str(i), **{field: value for field in CONTINUOUS})
            for i, value in enumerate([1., 3., 10000.])]
    transformed, stats = standardize_actions(rows, [0, 1])
    np.testing.assert_allclose(stats["mean"], 2.)
    np.testing.assert_allclose(stats["scale"], 1.)
    np.testing.assert_allclose(transformed[:2, 0], [-1., 1.])
    assert transformed[2, 0] == 9998.
    rows[2][CONTINUOUS[0]] = None
    with pytest.raises(ValueError, match="never be zero-imputed"):
        standardize_actions(rows, [0, 1])


def test_failure_detector_only_uses_explicit_outcomes():
    assert explicit_failure({"checks": [{"mechanical_grasp_detected": False}]})
    assert explicit_failure({"acquisition_status": "FAILED"})
    assert not explicit_failure({"failure_context": {"failures": []}})
    assert not explicit_failure({"mechanical_grasp_detected": None})
    assert not explicit_failure({"seed_outcome": "NO_PROGRESS"})


def test_motion_parser_reads_actual_close_lift_release_without_execution(tmp_path):
    motion = tmp_path / "motion.py"
    motion.write_text(
        'motion.move_tcp(left=[0.2,0.3,0.01,0,0,0])\n'
        'motion.gripper(arm="left", action="close")\n'
        'motion.move_tcp(left=[0.2,0.3,0.06,0,0,0])\n'
        'motion.move_tcp(left=[0.4,0.5,0.02,0,0,0])\n'
        'motion.gripper(arm="left", action="open")\n')
    pa, pb, lift, count = executed_geometry(motion)
    np.testing.assert_allclose(pa[:3], [200, 300, 10])
    np.testing.assert_allclose(pb[:3], [400, 500, 20])
    assert lift == pytest.approx(50.)
    assert count == 5
    motion.write_text('import os\nos.remove("never_execute")')
    with pytest.raises(ValueError):
        executed_geometry(motion)


def test_shuffle_never_uses_own_transition_and_is_reproducible():
    donors = shuffled_donors(20, 100, 42)
    assert np.all(donors != np.arange(20)[None, :])
    assert donors.min() >= 0 and donors.max() < 20
    np.testing.assert_array_equal(donors, shuffled_donors(20, 100, 42))
    with pytest.raises(ValueError):
        shuffled_donors(1, 100, 42)


def test_identity_start_and_metric_geometry():
    torch = pytest.importorskip("torch")
    z = torch.nn.functional.normalize(torch.randn(6, 384), dim=1)
    a = torch.randn(6, 9)
    for dim in [0, 9]:
        model = make_model(dim, 11)
        p = predict(model, z, a if dim else None)
        torch.testing.assert_close(z, p)
    metrics = evaluate(z.numpy(), z.numpy(), z.numpy(), ["a"]*3+["b"]*3)
    assert metrics["cosine_similarity"] == pytest.approx(1., abs=1e-6)
    assert metrics["latent_l2_error"] == 0.
    assert metrics["cosine_improvement_over_identity"] == 0.
