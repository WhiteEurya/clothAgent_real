"""Synthetic protocol tests, no network, encoder, or robot."""
import ast
import numpy as np
import pytest

from cloth_agent.experiments import teacher_experience_retrieval_probe_v1 as m


def references():
    return [dict(experience_id=f"s{i//2}/t{i}", trajectory_unit=f"s{i//2}") for i in range(8)]


def test_multi_citation_hit_recall_and_best_rank():
    refs = references()
    result = m.query_metrics(np.arange(8), refs, {"s0/t1", "s2/t4"})
    assert result["best_teacher_rank"] == 2
    assert result["reciprocal_rank"] == .5
    assert result["hit_at_1"] == 0
    assert result["hit_at_3"] == 1
    assert result["citation_recall_at_3"] == .5
    assert result["citation_recall_at_5"] == 1.


def test_diversity_does_not_relabel_same_session_or_shrink_denominator():
    refs = references()
    order = m.diverse_order(np.arange(8), refs)
    assert order.tolist() == [0, 2, 4, 6]
    result = m.query_metrics(order, refs, {"s0/t1", "s2/t4"})
    assert result["best_teacher_rank"] == 3
    assert result["citation_recall_at_20"] == .5
    missing = m.query_metrics(order, refs, {"s0/t1"})
    assert missing["best_teacher_rank"] is None
    assert missing["reciprocal_rank"] == 0
    assert missing["hit_at_20"] == 0
    aggregate = m.aggregate([dict(query_round_id="q0", **result),
                             dict(query_round_id="q1", **missing)])
    assert aggregate["mrr"] == pytest.approx(1/6)
    assert aggregate["best_teacher_rank"]["max"] is None
    assert aggregate["missing_teacher_queries"] == 1


def test_exact_ranking_stable_and_label_independent():
    vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    order, scores = m.exact_order(vectors, np.array([[1, 0]], dtype=np.float32))
    assert order.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(scores, [1, 1, 0, -1])


def test_reuse_only_pure_visual_definitions(tmp_path):
    tree = ast.parse(m.source_components())
    assert {node.name for node in tree.body} == {"DinoEncoder", "decode", "build_index"}
    text = m.source_components()
    assert "local_files_only=True" in text and "use_fast=False" in text
    assert "last_hidden_state[:,0]" in text
    assert "def trial(" not in text and "discovery_model" not in text
    path = tmp_path/"rgb.png"
    m.Image.new("RGB", (30, 20), "purple").save(path)
    namespace = dict(np=np, time=m.time, Path=m.Path, Image=m.Image, file_hash=m.sha)
    exec(compile(text, "test_reused_pipeline", "exec"), namespace)
    decoded = namespace["decode"](path)
    assert decoded.size == (30, 20)
    decoded.close()


def test_fixed_classification_boundaries():
    p = dict(query_count=7, hit={"5": 5/7, "20": 1.}, citation_recall={"5": .75, "20": 1.})
    assert m.classification(p) == "STRONG_RETRIEVAL_SIGNAL"
    p["citation_recall"]["5"] = .5
    assert m.classification(p) == "MODERATE_RETRIEVAL_SIGNAL"
    p["hit"]["20"] = 4/7
    assert m.classification(p) == "WEAK_RETRIEVAL_SIGNAL"
    assert m.classification(p, complete=False) == "UNRESOLVED"
