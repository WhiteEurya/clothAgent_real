"""Offline correctness checks; no model downloads or physical devices."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cloth_agent.dino_correspondence import (FeatureStore, evaluate, heatmap_overlay,
    image_metadata, input_to_original, match, original_to_input, patch_point,
    point_patch, polygon_distance, preprocess, separated_peaks, valid_patches)
from cloth_agent.dino_correspondence_tool import Experiment, ResultWriter, run_batch, summarize


def features(seed=7):
    f = np.random.default_rng(seed).normal(size=(1024, 384)).astype('float32')
    return f / np.linalg.norm(f, axis=1, keepdims=True)


@pytest.mark.parametrize('size', [(1280, 720), (641, 479), (479, 641), (1001, 333), (448, 448)])
def test_letterbox_roundtrip_and_pixel_centres(size):
    meta = image_metadata(size)
    tensor, actual = preprocess(Image.new('RGB', size, 'white'))
    assert actual == meta and tensor.shape == (3, 448, 448)
    for p in ([.5, .5], [size[0]-.5, size[1]-.5], [size[0]*.27, size[1]*.64]):
        assert input_to_original(original_to_input(p, meta), meta) == pytest.approx(p)
    for idx in np.flatnonzero(valid_patches(meta)):
        p = patch_point(int(idx), meta)
        assert 0 <= p[0] < size[0] and 0 <= p[1] < size[1]
        assert point_patch(p, meta) == idx
    left, top = meta['offset_xy']
    if top:
        assert np.all(tensor[:, :top] == 0)
    if left:
        assert np.all(tensor[:, :, :left] == 0)


def test_same_image_and_global_cycle():
    f = features(); meta = image_metadata((1280, 720))
    for point in ([400.5, 200.5], [640.5, 400.5], [1200.5, 600.5]):
        r, scores = match(f, f, meta, meta, point)
        assert r['candidates'][0]['index'] == point_patch(point, meta)
        assert r['top1_score'] == pytest.approx(1, abs=1e-6)
        assert r['cycle_error_patches'] == 0
        assert scores.shape == (32, 32)


def test_target_uses_its_own_inverse_mapping_and_reverse_search_is_global():
    a, b = features(1), features(2)
    ma, mb = image_metadata((1280, 720)), image_metadata((480, 800))
    source, target, far_source = 10*32+10, 20*32+17, 22*32+25
    b[target] = a[far_source]
    a[source] = .99*a[far_source]+.01*a[source]
    a[source] /= np.linalg.norm(a[source])
    r, _ = match(a, b, ma, mb, patch_point(source, ma))
    assert r['candidates'][0]['index'] == target
    assert r['candidates'][0]['xy'] == pytest.approx(patch_point(target, mb))
    assert r['back_index'] == far_source
    assert r['cycle_error_patches'] > 10


def test_padding_can_never_win_and_no_valid_candidate():
    a, b = features(1), features(2)
    meta = image_metadata((1280, 720)); source = 16*32+16
    b[0] = a[source]  # exact but pure-padding match must be excluded
    r, scores = match(a, b, meta, meta, patch_point(source, meta))
    assert all(c['index'] != 0 for c in r['candidates'])
    assert np.isneginf(scores[0, 0])
    r, scores = match(a, b, meta, image_metadata((10000, 1)), patch_point(source, meta))
    assert r['status'] == 'NO_VALID_TARGET_PATCH'
    assert not r['candidates'] and not np.isfinite(scores).any()


def test_nms_suppresses_neighbourhood_and_reports_separate_peaks():
    scores = np.full(1024, -np.inf, np.float32)
    for (r, c), value in [((10, 10), 1), ((10, 11), .99), ((10, 12), .98), ((10, 13), .9), ((25, 25), .8)]:
        scores[r*32+c] = value
    assert separated_peaks(scores) == [330, 333, 825]


def test_region_evaluation_is_separate_and_invisible_has_no_hit():
    f = features(); meta = image_metadata((448, 448))
    result, before = match(f, f, meta, meta, [147, 147])
    label = dict(visibility='visible', regions=[[[140, 140], [155, 140], [155, 155], [140, 155]]], before_prediction=True)
    e = evaluate(result, label, (448, 448))
    assert e['top1_hit'] and e['top1_region_error_px'] == 0
    label['regions'] = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
    assert evaluate(result, label, (448, 448))['top1_region_error_px'] == pytest.approx(np.sqrt(2)*137)
    _, after = match(f, f, meta, meta, [147, 147])
    assert np.array_equal(before, after)
    assert evaluate(result, dict(visibility='invisible', regions=[]), (448, 448))['top1_hit'] is None
    assert polygon_distance([10, 5], [[0, 0], [10, 0], [10, 10], [0, 10]]) == 0
    with pytest.raises(ValueError):
        evaluate(result, dict(visibility='visible', regions=[]), (448, 448))


def test_heatmap_uses_same_absolute_cosine_scale():
    image = Image.new('RGB', (448, 448), 'white'); meta = image_metadata(image.size)
    a = np.zeros((32, 32), np.float32); b = a.copy(); b[0, 0] = 1
    out_a, out_b = [np.asarray(heatmap_overlay(image, s, meta)) for s in (a, b)]
    assert np.array_equal(out_a[200, 200], out_b[200, 200])
    assert not np.array_equal(out_b[0, 0], out_b[200, 200])


class FakeEncoder:
    contract = {'test': 'normalized-patch-test-only'}
    load_s = 0.; device = 'cpu'
    def __init__(self):
        self.calls = 0
    def encode(self, images):
        self.calls += 1
        return np.stack([features() for _ in images]), dict(preprocess_s=0., forward_and_transfer_s=0.,
                                                           newly_encoded_images=len(images), forward_calls=1)


def test_cache_deduplicates_identical_pixels_and_invalidates_changed_image(tmp_path):
    path = tmp_path/'image.png'; Image.new('RGB', (640, 480), 'white').save(path)
    encoder = FakeEncoder(); store = FeatureStore(encoder, tmp_path/'cache')
    items, t = store.get_pair([path, path])
    assert t['newly_encoded_images'] == 1 and encoder.calls == 1
    store.get_pair([path, path]); assert encoder.calls == 1
    fresh = FeatureStore(encoder, tmp_path/'cache')
    _, t = fresh.get_pair([path, path]); assert encoder.calls == 1 and 'disk' in t['cache_sources']
    Image.new('RGB', (640, 480), 'black').save(path)
    fresh.get_pair([path, path]); assert encoder.calls == 2


def test_records_preserve_unlabeled_status_and_pair_level_counts(tmp_path):
    image = tmp_path/'image.png'; Image.new('RGB', (448, 448), 'white').save(image)
    exp = Experiment(tmp_path/'run'); exp.store = FeatureStore(FakeEncoder(), tmp_path/'cache')
    exp.load(image, image, 'p1')
    r, scores = exp.query([147, 147]); exp.persist(r, scores)
    label = dict(visibility='visible', regions=[[[140, 140], [155, 140], [155, 155], [140, 155]]], before_prediction=True)
    r, scores = exp.query([147, 147], label=label); exp.persist(r, scores)
    summary = summarize(exp.root)
    assert len(summary['per_pair']) == 1
    assert summary['total']['points'] == 2 and summary['total']['blindly_labeled_visible'] == 0
    assert summary['pair_macro_top1'] is None
    assert summary['diagnostic_distributions']['visible_hit']['count'] == 0
    assert summary['exploratory_diagnostic_distributions']['visible_hit']['count'] == 1
    assert (exp.root/'results.csv').exists()
    records = [json.loads(line) for line in (exp.root/'results.jsonl').read_text().splitlines()]
    assert records[0]['evaluation']['top1_hit'] is None
    with np.load(Path(records[0]['result_directory'])/'heatmap.npz') as data:
        assert data['cosine'].shape == (32, 32)


@pytest.fixture
def experiment(tmp_path):
    path = tmp_path/'image.png'
    Image.new('RGB', (448, 448), 'white').save(path)
    exp = Experiment(tmp_path/'run')
    exp.store = FeatureStore(FakeEncoder(), tmp_path/'cache')
    exp.load(path, path, 'pair')
    return exp, path


def visible_label():
    return dict(visibility='visible', before_prediction=True, target_xy=[150, 151],
                regions=[[[140, 140], [155, 140], [155, 155], [140, 155]]])


def test_reverse_query_uses_target_geometry_and_saves_source_target_order(tmp_path):
    a, b = tmp_path/'a.png', tmp_path/'b.png'
    Image.new('RGB', (1280, 720), 'white').save(a)
    Image.new('RGB', (480, 800), 'black').save(b)
    exp = Experiment(tmp_path/'run')
    exp.store = FeatureStore(FakeEncoder(), tmp_path/'cache')
    exp.load(a, b, 'asymmetric')
    # Source pixel is outside A's height but inside B: validates source selection.
    point = [240, 750]
    record, scores = exp.query(point, direction='B_to_A')
    expected, expected_scores = match(exp.pair[1]['features'], exp.pair[0]['features'],
                                    exp.pair[1]['meta'], exp.pair[0]['meta'], point)
    assert record['result']['candidates'] == expected['candidates']
    assert np.array_equal(scores, expected_scores)
    assert record['direction'] == 'B_to_A'
    assert record['images'][0]['meta']['original_wh'] == [480, 800]
    assert exp.pair[0]['image'].size == (1280, 720)  # UI A/B remain fixed
    assert exp.store.encoder.calls == 1
    exp.persist(record, scores)
    folder = Path(record['result_directory'])
    assert Image.open(folder/'source_cycle.png').size == (480, 800)
    assert Image.open(folder/'target_heatmap.png').size == (1280, 720)
    assert json.loads((folder/'annotation.json').read_text())['direction'] == 'B_to_A'
    assert summarize(exp.root)['per_direction']['B_to_A']['points'] == 1
    # Seeing either direction exposes the entire pair, including after reopening.
    reopened = Experiment(exp.root); reopened.store = exp.store
    reopened.load(a, b, 'renamed')
    record, _ = reopened.query([147, 147], label=visible_label())
    assert not record['evaluation']['eligible_for_blind_metrics']
    reopened.load(b, a, 'swapped')
    assert reopened.pair_exposed()
    with pytest.raises(ValueError, match='Direction'):
        exp.query(point, direction='invalid')


def test_exposure_survives_restart_without_artifact_save_and_pair_rename(experiment):
    exp, path = experiment
    record, _ = exp.query([147, 147], label=visible_label())
    assert record['evaluation']['eligible_for_blind_metrics']
    reopened = Experiment(exp.root)
    reopened.store = exp.store
    reopened.load(path, path, 'different_name')
    label = visible_label()
    record, _ = reopened.query([147, 147], label=label)
    assert label['before_prediction'] is True  # caller data not mutated
    assert not record['evaluation']['eligible_for_blind_metrics']
    assert record['label']['blind_check']['pair_previously_exposed']


def test_batch_freezes_all_labels_before_first_prediction_and_rejects_reuse(experiment):
    exp, path = experiment
    points = [dict(xy=[147, 147], label=visible_label()) for _ in range(2)]
    iterator = exp.query_many(points)
    first, scores = next(iterator)
    annotations = list((exp.root/'queries').glob('*/annotation.json'))
    assert len(annotations) == 2
    points[1]['label']['target_xy'] = [0, 0]
    second, scores2 = next(iterator)
    assert second['label']['target_xy'] == [150, 151]
    assert first['evaluation']['eligible_for_blind_metrics']
    assert second['evaluation']['eligible_for_blind_metrics']
    exp.persist(first, scores); exp.persist(second, scores2)
    case = dict(pair_id='renamed', a=path, b=path, points=points)
    summary = run_batch(exp, [case])
    assert summary['total']['blindly_labeled_visible'] == 2
    assert summary['diagnostic_distributions']['visible_hit']['count'] == 2
    assert summary['exploratory_diagnostic_distributions']['visible_hit']['count'] == 2


def test_exploratory_miss_and_invisible_are_separate_from_blind_distributions(experiment):
    exp, _ = experiment
    labels = [visible_label(), dict(visibility='invisible', before_prediction=True, regions=[])]
    for record, scores in exp.query_many([dict(xy=[147, 147], label=l) for l in labels]):
        exp.persist(record, scores)
    label = visible_label(); label['regions'] = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
    record, scores = exp.query([147, 147], label=label); exp.persist(record, scores)
    record, scores = exp.query([147, 147], label=labels[1]); exp.persist(record, scores)
    summary = summarize(exp.root)
    assert summary['diagnostic_distributions']['visible_miss']['count'] == 0
    assert summary['exploratory_diagnostic_distributions']['visible_miss']['count'] == 1
    assert summary['diagnostic_distributions']['invisible']['count'] == 1
    assert summary['exploratory_diagnostic_distributions']['invisible']['count'] == 1
    assert summary['total']['point_error_when_region_hit']['mean_px'] == pytest.approx(5)


def test_exact_point_error_is_distinct_from_region_and_cycle_error(experiment):
    exp, _ = experiment
    label = visible_label()
    record, scores = exp.query([147, 147], label=label)
    assert record['result']['cycle_error_patches'] == 0
    assert record['evaluation']['top1_region_error_px'] == 0
    assert record['evaluation']['top1_point_error_px'] == pytest.approx(5)
    label['target_xy'][0] = 300
    assert record['label']['target_xy'] == [150, 151]
    point_only = dict(visibility='visible', regions=[], target_xy=[150, 151], before_prediction=True)
    e = evaluate(record['result'], point_only, (448, 448))
    assert e['top1_hit'] is None and e['top1_point_error_px'] == pytest.approx(5)
    for target in ([448, 5], [-1, 2], [float('nan'), 2], [1]):
        with pytest.raises(ValueError):
            evaluate(record['result'], dict(point_only, target_xy=target), (448, 448))
    with pytest.raises(ValueError):
        evaluate(record['result'], dict(point_only, visibility='invisible'), (448, 448))


def test_point_only_labels_do_not_count_as_region_misses(experiment):
    exp, _ = experiment
    label = dict(visibility='visible', target_xy=[150, 151], before_prediction=True)
    record, scores = exp.query([147, 147], label=label); exp.persist(record, scores)
    summary = summarize(exp.root)
    assert summary['total']['blindly_labeled_visible'] == 1
    assert summary['total']['blindly_labeled_regions'] == 0
    assert summary['total']['top1_rate'] is None
    assert summary['total']['point_error']['mean_px'] == pytest.approx(5)
    assert summary['diagnostic_distributions']['visible_point_only']['count'] == 1


def test_no_prediction_counts_as_region_miss_but_has_no_point_error(experiment):
    exp, path = experiment
    thin = path.parent/'thin.png'
    Image.new('RGB', (10000, 1), 'white').save(thin)
    exp.load(path, thin)
    label = dict(visibility='visible', before_prediction=True, target_xy=[100, .5],
                 regions=[[[0, 0], [200, 0], [200, 1], [0, 1]]])
    record, scores = exp.query([147, 147], label=label); exp.persist(record, scores)
    summary = summarize(exp.root)
    assert summary['total']['top1_rate'] == 0
    assert summary['total']['point_error']['count'] == 0
    assert summary['total']['visible_without_prediction'] == 1
    assert summary['diagnostic_distributions']['visible_no_prediction']['count'] == 1


def test_async_saving_uses_snapshot_after_navigation_and_does_not_block(experiment, monkeypatch):
    import threading
    exp, path = experiment
    record, scores = exp.query([147, 147])
    writer = ResultWriter(exp)
    entered, release = threading.Event(), threading.Event()
    persist = exp.persist
    def delayed(record, scores):
        entered.set()
        assert release.wait(5)
        persist(record, scores)
    monkeypatch.setattr(exp, 'persist', delayed)
    try:
        writer.submit(record, scores)
        assert entered.wait(5)  # worker is blocked, but submit returned
        record['result']['source_xy'] = [0, 0]
        scores[:] = -1
        other = path.parent/'other.png'
        Image.new('RGB', (600, 400), 'black').save(other)
        exp.load(other, other, 'next_pair')
        release.set()
        for future in list(writer.pending):
            future.result(timeout=5)
        assert writer.poll() == [(record['query_id'], None)]
        folder = Path(record['result_directory'])
        saved = json.loads((folder/'result.json').read_text())
        assert saved['pair_id'] == 'pair' and saved['result']['source_xy'] == [147, 147]
        with Image.open(folder/'target_candidates.png') as im:
            assert im.size == (448, 448) and im.getpixel((400, 400)) == (255, 255, 255)
        with np.load(folder/'heatmap.npz') as data:
            assert data['cosine'].max() > .99
    finally:
        release.set(); writer.shutdown()


def test_save_failure_retry_is_reported_and_does_not_duplicate_indexes(experiment, monkeypatch):
    import csv
    exp, _ = experiment
    record, scores = exp.query([147, 147], label=visible_label())
    writer = ResultWriter(exp)
    persist = exp.persist
    def fail_after_save(*args):
        persist(*args)
        raise OSError('simulated disk error')
    monkeypatch.setattr(exp, 'persist', fail_after_save)
    try:
        writer.submit(record, scores)
        future = next(iter(writer.pending))
        with pytest.raises(OSError):
            future.result(timeout=5)
        assert 'simulated disk error' in writer.poll()[0][1]
        assert len(writer.failed) == 1
        monkeypatch.setattr(exp, 'persist', persist)
        writer.retry()
        for future in list(writer.pending):
            future.result(timeout=5)
        assert writer.poll() == [(record['query_id'], None)]
        assert not writer.failed
        assert len((exp.root/'results.jsonl').read_text().splitlines()) == 1
        with (exp.root/'results.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1 and float(rows[0]['top1_point_error_px']) == pytest.approx(5)
    finally:
        writer.shutdown()
