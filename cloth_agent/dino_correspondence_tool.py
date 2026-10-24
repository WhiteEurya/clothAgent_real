"""Local desktop/batch entrypoint for frozen DINOv2 correspondence diagnostics."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .dino_correspondence import (DEFAULT_MODEL, FeatureStore, FrozenDino, draw_result,
    evaluate, heatmap_overlay, match, point_patch, save_json, sha, validate_label)


def utc():
    return datetime.now(timezone.utc).isoformat()


class Experiment:
    def __init__(self, output, model_path=DEFAULT_MODEL, device='auto'):
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_path, self.device = model_path, device
        self.store = None
        self.pair = None
        self.exposed = set()
        # Include legacy runs and queries whose artifact saving was interrupted.
        result_file = self.root/'results.jsonl'
        if result_file.exists():
            for line in result_file.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.exposed.add(tuple(sorted(i['rgb_pixels_sha256'] for i in row['images'])))
        for path in (self.root/'exposures').glob('*.json'):
            self.exposed.add(tuple(sorted(json.loads(path.read_text())['pair_key'])))

    def pair_key(self):
        return tuple(sorted(i['rgb_pixels_sha256'] for i in self.pair))

    def pair_exposed(self):
        return self.pair is not None and self.pair_key() in self.exposed

    def load(self, a, b, pair_id=None, group='UNASSIGNED'):
        if self.store is None:
            self.store = FeatureStore(FrozenDino(self.model_path, self.device), self.root/'features')
            save_json(self.root/'encoder.json', self.store.encoder.contract)
        self.pair, self.timing = self.store.get_pair([a, b])
        self.pair_id = pair_id or 'pair_' + '_'.join(x['rgb_pixels_sha256'][:10] for x in self.pair)
        self.group = group
        # Immutable clean RGB copies are the exact decoded inputs, never overlays.
        for item in self.pair:
            clean = self.root/'images'/(item['rgb_pixels_sha256']+'.png')
            clean.parent.mkdir(exist_ok=True)
            if not clean.exists():
                item['image'].save(clean)
            item['clean_rgb'] = str(clean.relative_to(self.root))
        return self.pair, self.timing

    def query(self, point, radius=2., label=None, point_name='', group=None, direction='A_to_B'):
        prepared = self._prepare_query(point, radius, label, point_name, group, direction)
        return self._run_prepared(prepared)

    def query_many(self, points):
        """Freeze every label for this pair before computing the first prediction."""
        prepared = [self._prepare_query(p['xy'], label=p.get('label'),
                                       point_name=p.get('name', ''), direction=p.get('direction', 'A_to_B'))
                    for p in deepcopy(points)]
        for item in prepared:
            yield self._run_prepared(item)

    def _prepare_query(self, point, radius=2., label=None, point_name='', group=None, direction='A_to_B'):
        if self.pair is None:
            raise ValueError('Load two images first')
        if direction not in ('A_to_B', 'B_to_A'):
            raise ValueError('Direction must be A_to_B or B_to_A')
        pair = self.pair if direction == 'A_to_B' else self.pair[::-1]
        a, b = pair
        label = deepcopy(label or dict(visibility='unlabeled', regions=[], before_prediction=False))
        # Validate/save annotation before running a search; never pass it into match.
        validate_label(label, b['image'].size)
        point_patch(point, a['meta'])
        if not math.isfinite(radius) or radius < 0:
            raise ValueError('NMS radius must be nonnegative and finite')
        requested_blind = label.get('before_prediction') is True
        exposed = self.pair_exposed()
        label['before_prediction'] = requested_blind and not exposed
        label['blind_check'] = dict(requested=requested_blind, pair_previously_exposed=exposed,
                                   scope='this output directory', checked_at=utc())
        query_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'_'+uuid.uuid4().hex[:6]
        folder = self.root/'queries'/query_id
        folder.mkdir(parents=True, exist_ok=False)
        save_json(folder/'annotation.json', dict(label=label, source_xy=list(point),
            pair_id=self.pair_id, direction=direction, point_name=point_name, recorded_at=utc()))
        record = dict(protocol='dino-point-correspondence-v1', query_id=query_id,
            pair_id=self.pair_id, group=group or self.group, point_name=point_name, created_at=utc(),
            direction=direction, image_order='source_target',
            images=deepcopy([{k: v for k, v in x.items() if k not in ('image', 'features')} for x in pair]),
            label=label, timing=dict(self.timing),
            encoder_contract=self.store.encoder.contract,
            label_never_used_for_search=True, result_directory=str(folder))
        return record, pair, list(point), radius

    def _run_prepared(self, prepared):
        record, pair, point, radius = prepared
        a, b = pair
        key = tuple(sorted(i['rgb_pixels_sha256'] for i in pair))
        # Durable before returning a prediction, even if later PNG/CSV saving fails.
        save_json(self.root/'exposures'/(record['query_id']+'.json'),
                  dict(pair_key=key, query_id=record['query_id'], recorded_at=utc()))
        self.exposed.add(key)
        result, scores = match(a['features'], b['features'], a['meta'], b['meta'], point, radius)
        record.update(result=result, evaluation=evaluate(result, record['label'], b['image'].size))
        record['timing']['matching_s'] = result['matching_s']
        return record, scores

    def persist(self, record, scores):
        started = time.perf_counter()
        folder = Path(record['result_directory'])
        # Resolve the immutable inputs from this query, never the active UI pair.
        a, b = record['images']
        with Image.open(self.root/a['clean_rgb']) as im:
            image_a = im.convert('RGB')
        with Image.open(self.root/b['clean_rgb']) as im:
            image_b = im.convert('RGB')
        np.savez_compressed(folder/'heatmap.npz', cosine=scores,
                            valid=np.isfinite(scores), color_limits=np.array([-1., 1.]))
        overlay = heatmap_overlay(image_b, scores, b['meta'])
        draw_result(overlay, record['result'], record['label']).save(folder/'target_heatmap.png')
        draw_result(image_b, record['result'], record['label']).save(folder/'target_candidates.png')
        draw_result(image_a, record['result'], source=True).save(folder/'source_cycle.png')
        record['timing']['artifact_save_s'] = time.perf_counter()-started
        save_json(folder/'result.json', record)
        # Rebuild small indexes atomically: retries cannot append duplicate rows,
        # and existing v1 CSVs gain the new optional point-error columns safely.
        target = self.root/'results.jsonl'
        records_by_id = {}
        if target.exists():
            for line in target.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    records_by_id[row['query_id']] = row
        for p in sorted((self.root/'queries').glob('*/result.json')):
            row = json.loads(p.read_text())
            records_by_id[row['query_id']] = row
        records = sorted(records_by_id.values(), key=lambda r: r['query_id'])
        temp = target.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
        temp.replace(target)
        csv_path = self.root/'results.csv'; temp = csv_path.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_row(record)))
            writer.writeheader()
            writer.writerows(csv_row(r) for r in records)
        temp.replace(csv_path)


class ResultWriter:
    """Single background writer; owns independent snapshots and retryable failures."""
    def __init__(self, experiment):
        self.experiment = experiment
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dino-save')
        self.pending = {}
        self.failed = {}

    def submit(self, record, scores):
        snapshot = deepcopy(record), scores.copy()
        future = self.executor.submit(self.experiment.persist, *snapshot)
        self.pending[future] = snapshot

    def poll(self):
        events = []
        for future, snapshot in list(self.pending.items()):
            if not future.done():
                continue
            del self.pending[future]
            query_id = snapshot[0]['query_id']
            try:
                future.result()
                self.failed.pop(query_id, None)
                events.append((query_id, None))
            except Exception as exc:
                self.failed[query_id] = snapshot
                events.append((query_id, str(exc)))
        return events

    def retry(self):
        failed, self.failed = self.failed, {}
        for snapshot in failed.values():
            self.submit(*snapshot)

    def shutdown(self):
        self.executor.shutdown(wait=True)


def csv_row(record):
    r, e, t = record['result'], record['evaluation'], record['timing']
    return dict(query_id=record['query_id'], pair_id=record['pair_id'], group=record['group'],
        direction=record.get('direction', 'A_to_B'),
        point_name=record['point_name'], source_x=r['source_xy'][0], source_y=r['source_xy'][1],
        visibility=e['visibility'], before_prediction=e['eligible_for_blind_metrics'],
        top1_hit=e['top1_hit'], top3_hit=e['top3_hit'], top1_region_error_px=e['top1_region_error_px'],
        target_xy_json=json.dumps(record['label'].get('target_xy')),
        top1_point_error_px=e.get('top1_point_error_px'),
        candidates_json=json.dumps(r['candidates']), top1_score=r['top1_score'],
        independent_peak_gap=r['independent_peak_gap'], cycle_error_patches=r['cycle_error_patches'],
        model_load_s=t['model_load_s'], pair_load_s=t['pair_load_s'],
        preprocess_s=t['preprocess_s'], encode_s=t['forward_and_transfer_s'],
        newly_encoded_images=t['newly_encoded_images'], matching_s=t['matching_s'],
        click_to_draw_s=t.get('click_to_draw_s'), artifact_save_s=t.get('artifact_save_s'), status=r['status'],
        result_directory=record['result_directory'])


def summarize(root):
    root = Path(root)
    records = [json.loads(line) for line in (root/'results.jsonl').read_text().splitlines() if line.strip()]
    pairs = {}
    for r in records:
        pairs.setdefault(r['pair_id'], []).append(r)
    def metrics(rows):
        visible = [r for r in rows if r['evaluation']['eligible_for_blind_metrics'] and
                   r['evaluation']['visibility'] == 'visible']
        region_rows = [r for r in visible if r['label'].get('regions')]
        n = len(region_rows)
        point_errors = [r['evaluation']['top1_point_error_px'] for r in visible
                        if r['evaluation'].get('top1_point_error_px') is not None]
        correct_part_errors = [r['evaluation']['top1_point_error_px'] for r in region_rows
                               if r['evaluation']['top1_hit'] is True and
                               r['evaluation'].get('top1_point_error_px') is not None]
        def error_stats(values):
            return dict(count=len(values), mean_px=float(np.mean(values)) if values else None,
                        median_px=float(np.median(values)) if values else None,
                        p90_px=float(np.percentile(values, 90)) if values else None)
        return dict(points=len(rows), blindly_labeled_visible=len(visible),
            blindly_labeled_regions=n,
            top1_hits=sum(r['evaluation']['top1_hit'] is True for r in visible),
            top3_hits=sum(r['evaluation']['top3_hit'] is True for r in visible),
            top1_rate=None if not n else sum(r['evaluation']['top1_hit'] is True for r in region_rows)/n,
            top3_rate=None if not n else sum(r['evaluation']['top3_hit'] is True for r in region_rows)/n,
            point_error=error_stats(point_errors),
            point_error_when_region_hit=error_stats(correct_part_errors),
            visible_without_prediction=sum(not r['result']['candidates'] for r in visible),
            invisible=sum(r['label']['visibility'] == 'invisible' for r in rows),
            uncertain=sum(r['label']['visibility'] == 'uncertain' for r in rows))
    per_pair = {key: metrics(rows) for key, rows in pairs.items()}
    rates = [v['top1_rate'] for v in per_pair.values() if v['top1_rate'] is not None]
    def distributions(blind):
        buckets = {key: [] for key in ('visible_hit', 'visible_miss', 'visible_point_only',
                   'visible_no_prediction', 'invisible', 'uncertain', 'unlabeled')}
        for r in records:
            e = r['evaluation']
            if bool(e['eligible_for_blind_metrics']) != blind:
                continue
            bucket = e['visibility']
            if bucket == 'visible':
                if not r['result']['candidates']:
                    bucket = 'visible_no_prediction'
                elif e['top1_hit'] is None:
                    bucket = 'visible_point_only'
                else:
                    bucket = 'visible_hit' if e['top1_hit'] else 'visible_miss'
            buckets[bucket].append(r)
        return {bucket: dict(count=len(rows), **{key: [r['result'][key] for r in rows
                            if r['result'][key] is not None]
                            for key in ('top1_score', 'independent_peak_gap', 'cycle_error_patches')})
                for bucket, rows in buckets.items()}
    summary = dict(total=metrics(records), per_pair=per_pair,
        per_direction={direction: metrics([r for r in records if r.get('direction', 'A_to_B') == direction])
                       for direction in ('A_to_B', 'B_to_A')},
        pair_macro_top1=None if not rates else sum(rates)/len(rates),
        diagnostic_distributions=distributions(True),
        exploratory_diagnostic_distributions=distributions(False),
        diagnostic_distributions_scope='blind labels only; exploratory labels are separate',
        interpretation='Repeated points on a pair are correlated; no probability calibration or automatic rejection rule. '
                       'Region distance is not precise physical-point error. Human invisible/uncertain labels have no localization hit metric.')
    save_json(root/'summary.json', summary)
    return summary


def prepare_cases(graph_path, current_root, output, count=16):
    """Select real clean images without using DINO outcomes or claiming GT."""
    if count < 8:
        raise ValueError('--count must be at least 8 for this seed-pool recipe')
    graph_path = Path(graph_path).resolve(); graph = json.loads(graph_path.read_text())
    source = graph['source_trajectory']; trajectory_path = Path(source['path'])
    if sha(trajectory_path) != source['sha256']:
        raise ValueError('Trajectory hash differs from graph')
    trajectory = json.loads(trajectory_path.read_text())
    observations = trajectory['observations']
    reference_nodes = {oid: node['node_id'] for node in graph['nodes'] for oid in node['representative_observations']}
    cases = []; used = set()
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    def rgb(oid):
        ref = observations[oid]['rgb']; path = Path(ref['path'])
        if sha(path) != ref['sha256']:
            raise ValueError('Graph RGB hash mismatch')
        return path
    def add(a, b, selection, oid, point=None, self_check=False):
        key = (sha(a), sha(b))
        if key in used:
            return
        used.add(key)
        case_id = f'pair_{len(cases):02d}'
        paths = []
        for role, path in [('A', a), ('B', b)]:
            target = root/'images'/f'{case_id}_{role}.png'; target.parent.mkdir(exist_ok=True)
            with Image.open(path) as im:
                im.convert('RGB').save(target)
            paths.append(str(target.relative_to(root)))
        cases.append(dict(pair_id=case_id, a=paths[0], b=paths[1], group='SELF' if self_check else 'UNASSIGNED',
            selection=selection, source_graph_node=reference_nodes[oid], source_observation=oid,
            original_paths=[str(a.resolve()), str(b.resolve())],
            file_sha256=[sha(root/p) for p in paths], suggested_source_xy=point,
            label=dict(visibility='unlabeled', regions=[], before_prediction=False),
            note='Candidate pool only. Human assigns A/B/C/D and labels correspondence before viewing results. '
                 'No executed B grasp is used as ground truth. Suggested A point is a recorded start, not verified acquisition.'))
    steps = [s for s in trajectory['steps'] if s['before_observation'] in reference_nodes
             and s.get('after_observation') in observations]
    def anchor(step):
        pts = step['commanded_action'].get('selected_points', [])
        p = next((v['pixel_xy'] for v in pts if v['name'] == 'start'), None)
        return None if p is None else [float(p[0])+.5, float(p[1])+.5]
    first = steps[0]; oid = first['before_observation']
    add(rgb(oid), rgb(oid), 'same-image coordinate check', oid, anchor(first), True)
    # Existing same-node labels are model-produced hints, never correspondence GT.
    for node in graph['nodes']:
        refs = node['representative_observations']
        if len(refs) >= 2:
            add(rgb(refs[0]), rgb(refs[-1]), 'same graph node; human state verification pending', refs[0])
        if len(cases) >= 5:
            break
    for index in np.linspace(0, len(steps)-1, 4, dtype=int):
        step = steps[index]; oid = step['before_observation']
        add(rgb(oid), rgb(step['after_observation']), 'recorded before/after; target visibility unknown', oid, anchor(step))
    fresh = sorted(Path(current_root).glob('**/round_*/before/rgb.png'))
    if not fresh:
        raise ValueError('No clean round_*/before/rgb.png found in --current-root')
    remaining = max(0, count+1-len(cases))
    for k, index in enumerate(np.linspace(0, len(fresh)-1, remaining, dtype=int)):
        step = steps[(k*max(1, len(steps)//max(1, remaining))) % len(steps)]
        oid = step['before_observation']
        add(rgb(oid), fresh[index], 'graph reference vs later episode RGB; correspondence unknown', oid, anchor(step))
    manifest = dict(protocol='dino-point-cases-v1', graph=str(graph_path), graph_sha256=sha(graph_path),
                    created_at=utc(), cases=cases, labels_human_verified=False,
                    missing_coverage='Different-garment group C and occluded group D require human verification/additional images.')
    save_json(root/'cases.json', manifest)
    # Small contact sheet contains only clean images and outside-image captions.
    sheet = Image.new('RGB', (640, len(cases)*145), '#162130'); draw = ImageDraw.Draw(sheet)
    for i, case in enumerate(cases):
        draw.text((8, i*145+3), case['pair_id']+'  '+case['group'], fill='white')
        for col, key in enumerate(('a', 'b')):
            with Image.open(root/case[key]) as im:
                im.thumbnail((310, 120)); sheet.paste(im, (col*320+5, i*145+22))
    sheet.save(root/'contact_sheet.jpg')
    return root/'cases.json'


def load_cases(path):
    path = Path(path).resolve(); data = json.loads(path.read_text())
    cases = data['cases']
    if len({c['pair_id'] for c in cases}) != len(cases):
        raise ValueError('Pair IDs must be unique')
    for c in cases:
        for i, key in enumerate(('a', 'b')):
            c[key] = str((path.parent/c[key]).resolve())
            if c.get('file_sha256') and sha(c[key]) != c['file_sha256'][i]:
                raise ValueError('Case RGB changed')
    return cases


def run_batch(experiment, cases):
    for case in cases:
        experiment.load(case['a'], case['b'], case['pair_id'], case.get('group', 'UNASSIGNED'))
        points = case.get('points', [])
        if not points:
            # Smoke clicks are explicitly unlabeled. They are not semantic tests.
            w, h = experiment.pair[0]['image'].size
            points = [dict(xy=case.get('suggested_source_xy') or [w*.5, h*.5], name='unlabeled_smoke')]
        for record, scores in experiment.query_many(points):
            experiment.persist(record, scores)
        print(case['pair_id'], record['result']['status'], flush=True)
    return summarize(experiment.root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('ui', 'batch'):
        p = sub.add_parser(name)
        p.add_argument('--cases', type=Path, required=name == 'batch')
        p.add_argument('--a', type=Path); p.add_argument('--b', type=Path)
        p.add_argument('--output', type=Path, default=Path('reports/dino_correspondence'))
        p.add_argument('--model-path', type=Path, default=DEFAULT_MODEL)
        p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p = sub.add_parser('prepare')
    p.add_argument('--graph', type=Path, required=True); p.add_argument('--current-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--count', type=int, default=16)
    p = sub.add_parser('summarize'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        print(prepare_cases(args.graph, args.current_root, args.output, args.count)); return
    if args.command == 'summarize':
        print(json.dumps(summarize(args.output), ensure_ascii=False, indent=2)); return
    experiment = Experiment(args.output, args.model_path, args.device)
    cases = load_cases(args.cases) if args.cases else []
    if (args.a is None) != (args.b is None):
        parser.error('--a and --b must be supplied together')
    if args.a:
        cases.insert(0, dict(pair_id=f'custom_{sha(args.a)[:10]}_{sha(args.b)[:10]}',
                             a=str(args.a.resolve()), b=str(args.b.resolve()), group='UNASSIGNED'))
    if args.command == 'batch':
        print(json.dumps(run_batch(experiment, cases)['total'], ensure_ascii=False)); return
    from .dino_correspondence_ui import launch
    launch(experiment, cases)


if __name__ == '__main__':
    main()
