"""Isolated, offline action-conditioned DINOv2 CLS dynamics diagnostic.

Stages: audit, encode, train, report. No project runtime imports, network models,
robot interfaces, RGB prediction, planning, or adaptive model selection.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


VERSION = "world_model_probe_v1"
MODEL_PATH = Path("/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small")
SEEDS = [11, 23, 37, 53, 71]
FRACTIONS = [.2, .4, .6, .8, 1.]
CONTINUOUS = ["pa_x", "pa_y", "pb_x", "pb_y", "dx", "dy", "distance", "executed_lift_mm"]
OPTIONAL = ["lift_height_mm", "grasp_offset_mm", "clearance_mm", "place_offset_mm",
            "speed_mm_s", "approach_height_mm", "approach_rpy_deg"]
PROTOCOL = {
    "version": VERSION, "seeds": SEEDS, "fractions": FRACTIONS, "split_seed": 20261020,
    "hidden_widths": [256, 256], "activation": "GELU", "optimizer": "AdamW",
    "learning_rate": .001, "weight_decay": .01, "max_epochs": 600,
    "patience": 60, "min_delta": 1e-6, "batch_size": "full training subset",
    "early_stopping": "validation group-macro cosine; epoch zero is an eligible checkpoint",
    "loss": "mean(1 - cosine), equal transition weights",
    "shuffle_repeats": 100, "bootstrap_repeats": 10000,
    "stability_rule": "positive C-B in >=4/5 seeds AND >=2/3 held-out groups AND "
                      "positive lower 95% paired group-bootstrap CI; mean gain >=0.001",
    "shuffle_rule": "real-shuffled mean gain >=0.001, positive in >=4/5 seeds, "
                    "positive lower 95% group-bootstrap CI",
    "curve_rule": "100%-80% C gain >=0.001, positive in >=4/5 seeds, "
                  "positive lower 95% paired group-bootstrap CI",
    "classification": "LEARNABLE_SIGNAL if stable C>B, robust C>Identity, and shuffle rule; "
                      "otherwise DATA_LIMITED if curve rule; otherwise UNRESOLVED. "
                      "REPRESENTATION_LIMITED requires independent evidence this probe cannot supply.",
    "no_test_tuning": True,
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pixel_info(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.size, hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def explicit_failure(value):
    """Only structured outcome evidence; no substring matching warning templates."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("mechanical_grasp_detected", "grasp_success", "acquisition_success",
                       "cloth_acquired", "cloth_manipulated") and item is False:
                return True
            if key in ("grasp_status", "acquisition_status", "outcome") and isinstance(item, str):
                if item.upper() in ("FAILED", "FAILURE", "MISSED_GRASP", "NO_ACQUISITION"):
                    return True
            if explicit_failure(item):
                return True
    elif isinstance(value, list):
        return any(explicit_failure(v) for v in value)
    return False


def executed_geometry(path):
    """Read literal motion arguments using AST, NEVER execute recorded motion.py."""
    calls = []
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("non-literal recorded motion")
        call = node.value
        if not isinstance(call.func, ast.Attribute):
            raise ValueError("unexpected recorded motion call")
        calls.append((call.func.attr, {k.arg: ast.literal_eval(k.value) for k in call.keywords}))
    closes = [i for i, (name, kw) in enumerate(calls) if name == "gripper" and kw.get("action") == "close"]
    if len(closes) != 1:
        raise ValueError("not exactly one recorded acquisition")
    close = closes[0]
    opens = [i for i, (name, kw) in enumerate(calls) if i > close and name == "gripper" and kw.get("action") == "open"]
    if len(opens) != 1:
        raise ValueError("not exactly one recorded release")
    before = [kw["left"] for name, kw in calls[:close] if name == "move_tcp"]
    held = [kw["left"] for name, kw in calls[close:opens[0]] if name == "move_tcp"]
    if not before or not held or not np.isfinite(before + held).all():
        raise ValueError("missing recorded acquisition/release TCP")
    return np.array(before[-1]) * np.array([1000]*3 + [1]*3), \
        np.array(held[-1]) * np.array([1000]*3 + [1]*3), \
        (max(p[2] for p in held) - before[-1][2]) * 1000, len(calls)


def group_sessions(rows, links):
    parent = {r["session"]: r["session"] for r in rows}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for a, b in links:
        if a in parent and b in parent:
            x, y = sorted((root(a), root(b)))
            parent[y] = x
    # Exact duplicated RGBs also connect sessions, before splitting.
    owners = {}
    for row in rows:
        for key in ("before_pixel_sha256", "after_pixel_sha256"):
            other = owners.setdefault(row[key], row["session"])
            x, y = sorted((root(other), root(row["session"])))
            parent[y] = x
    for row in rows:
        row["group"] = root(row["session"])


def assign_split(rows, seed):
    groups = sorted({r["group"] for r in rows})
    if len(groups) < 7:
        raise ValueError("insufficient independent groups for train/validation/test diagnostic")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    ntest = max(2, math.ceil(.2 * len(groups)))
    nval = max(2, math.ceil(.15 * len(groups)))
    # Balance sample counts using group sizes only, before computing embeddings.
    # The seeded order breaks ties. Never use actions, targets or model scores.
    counts = Counter(r["group"] for r in rows)
    target = .2 * len(rows)
    best = None
    for test in itertools.combinations(groups, ntest):
        rest = [g for g in groups if g not in test]
        for validation in itertools.combinations(rest, nval):
            cost = (sum(counts[g] for g in test)-target)**2 + (sum(counts[g] for g in validation)-target)**2
            if best is None or cost < best[0]:
                best = (cost, test, validation)
    _, test, validation = best
    splits = {"test": list(test), "validation": list(validation),
              "train": [g for g in groups if g not in test and g not in validation]}
    for row in rows:
        row["split"] = next(k for k, g in splits.items() if row["group"] in g)
    return splits


def audit(root, output, scope_dates):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "data_audit.json").exists():
        raise ValueError("audit already frozen; use another output directory")
    rows, excluded, candidates = [], [], []
    source_hashes, links = {}, []
    # Conservative union of ALL documented cross-session interrupted-run boundaries,
    # regardless of visual-state equivalence. Graph labels never enter the model.
    for p in (Path("reports/arm6_graph_incremental_20261019_v3/audit.json"),
              Path("reports/arm6_graph_incremental_20261020_v4/audit.json")):
        if p.exists():
            source_hashes[str(p.resolve())] = sha(p)
            for boundary in read(p)["interrupted_run_boundaries"]:
                a, b = [Path(boundary[k]).parent.name for k in ("from_trajectory", "to_trajectory")]
                if a != b:
                    links.append((a, b))
    for path in sorted(root.glob("*/**/traj.json")):
        rel = path.relative_to(root)
        session = rel.parts[0]
        manifest = read(path)
        reason = None
        if "resets" in rel.parts or manifest.get("not_a_demonstration") or manifest.get("collection_mode") == "random_reset":
            reason = "reset"
        elif "pending" in rel.parts:
            reason = "pending"
        elif session[:8] not in scope_dates:
            reason = "outside_confirmed_unfolding_session_scope"
        elif manifest.get("operator_accepted") is not True or manifest.get("status") != "COMPLETE":
            reason = "not_accepted_complete"
        elif manifest.get("arm") != "left" or manifest.get("camera") != "A":
            reason = "not_arm6_camera_A"
        folder = path.parent
        record = dict(id=f"{session}/{folder.name}", session=session,
                      trajectory=folder.name, source_folder=str(folder.resolve()))
        if reason:
            excluded.append(dict(**record, reason=reason))
            continue
        fields = {}
        points_path = folder / "points.json"
        points = read(points_path) if points_path.exists() else []
        by_name = {p["name"]: p for p in points}
        for name, prefix in (("start", "pa"), ("target", "pb")):
            point = by_name.get(name, {})
            fields[prefix + "_pixel_xy"] = point.get("pixel_xy")
            fields[prefix + "_world_xyz_mm"] = point.get("world_xyz_mm")
        fields.update({k: manifest.get("parameters", {}).get(k) for k in OPTIONAL})
        fields["primitive"] = manifest.get("collection_mode")
        evidence_paths = [path, points_path, folder/"events.jsonl", folder/"states.jsonl"]
        result_paths = list(folder.glob("*/result.json"))
        events = jsonl(folder / "events.jsonl")
        candidates.append(dict(**record, action_fields=fields))
        try:
            if explicit_failure(manifest) or explicit_failure(events) or any(explicit_failure(read(p)) for p in result_paths):
                raise ValueError("explicit_grasp_or_acquisition_failure")
            mode = manifest.get("collection_mode")
            if mode != "automatic_pick_place":
                raise ValueError("assisted_actual_grasp_not_representable_by_selected_Pa")
            result_path = folder / "automatic_pick_place/result.json"
            result = read(result_path)
            if result.get("status") != "EXECUTED" or result.get("physical_execution") is not True:
                raise ValueError("action_not_physically_executed")
            motion = folder / "automatic_pick_place/motion.py"
            pa, pb, lift, commands = executed_geometry(motion)
            if result.get("completed_commands") != commands:
                raise ValueError("incomplete_executed_motion")
            if not all(k in by_name for k in ("start", "target")):
                raise ValueError("missing_actual_action_parameters")
            for key, tcp in (("start", pa), ("target", pb)):
                if not np.allclose(np.array(by_name[key]["world_xyz_mm"])[:2], tcp[:2], atol=.01):
                    raise ValueError("selected_points_disagree_with_executed_motion")
            samples = jsonl(folder / "states.jsonl")
            if not samples or len(samples) != manifest["samples"] or not np.isfinite([s["world_tcp_mm_deg"] for s in samples]).all():
                raise ValueError("invalid_measured_execution_trace")
            before, after = folder / "observation/rgb.png", folder / "after/rgb.png"
            if not before.exists() or not after.exists():
                raise ValueError("missing_dedicated_before_or_after")
            size, before_hash = pixel_info(before)
            after_size, after_hash = pixel_info(after)
            if after_size != size:
                raise ValueError("before_after_resolution_mismatch")
            for obs in (folder/"observation/observation.json", folder/"after/observation.json"):
                meta = read(obs)
                if meta.get("synthetic"):
                    raise ValueError("synthetic_observation")
                evidence_paths.append(obs)
            ax, ay = by_name["start"]["pixel_xy"]
            bx, by = by_name["target"]["pixel_xy"]
            w, h = size
            if not (0 <= ax < w and 0 <= bx < w and 0 <= ay < h and 0 <= by < h):
                raise ValueError("action_pixel_outside_image")
            ax, bx, ay, by = ax/(w-1), bx/(w-1), ay/(h-1), by/(h-1)
            action = dict(pa_x=ax, pa_y=ay, pb_x=bx, pb_y=by, dx=bx-ax, dy=by-ay,
                          distance=math.hypot(bx-ax, by-ay), executed_lift_mm=lift)
            if not np.isfinite(list(action.values())).all():
                raise ValueError("nonfinite_action")
            evidence_paths += [before, after, motion, *result_paths]
            for p in evidence_paths:
                source_hashes[str(p.resolve())] = sha(p)
            rows.append(dict(**record, episode_id=manifest.get("episode_id"),
                             before_rgb=str(before.resolve()), after_rgb=str(after.resolve()),
                             before_pixel_sha256=before_hash, after_pixel_sha256=after_hash,
                             before_sha256=sha(before), after_sha256=sha(after),
                             executed_action=str(motion.resolve()), action_result=str(result_path.resolve()),
                             primitive=mode, **action))
        except (ValueError, KeyError, OSError) as exc:
            excluded.append(dict(**record, reason=str(exc)))
    if not rows:
        raise ValueError("no usable transitions")
    group_sessions(rows, links)
    splits = assign_split(rows, PROTOCOL["split_seed"])
    coverage = {}
    for key in candidates[0]["action_fields"]:
        n = sum(r["action_fields"][key] is not None for r in candidates)
        coverage[key] = dict(present=n, denominator=len(candidates), fraction=n/len(candidates))
    for key in CONTINUOUS:
        coverage[key] = dict(present=len(rows), denominator=len(rows), fraction=1.,
                             population="final usable automatic executions", included=True)
    for key in OPTIONAL:
        coverage[key]["included"] = False
        coverage[key]["reason"] = ("missing/low coverage; never imputed" if coverage[key]["fraction"] < .9
                                  else "execution-equivalent lift derived from motion.py" if key == "lift_height_mm"
                                  else "not needed in fixed minimal action vector; configured defaults "
                                       "not treated as independently measured execution parameters")
    coverage["primitive"]["included"] = True
    coverage["primitive"]["encoding"] = "one-hot; fixed automatic_pick_place category (constant)"
    review_path = output / "review/assessment.json"
    if review_path.exists():
        review = read(review_path)
        if set(review["reviewed_ids"]) != {r["id"] for r in candidates}:
            raise ValueError("visual review does not match current candidate snapshot")
        if review["excluded_ids"]:
            raise ValueError("review exclusions require explicit removal before freezing this probe")
        source_hashes[str(review_path.resolve())] = sha(review_path)
        review["candidate_pairs_reviewed"] = len(review["reviewed_ids"])
    else:
        review = dict(candidate_pairs_reviewed=0, status="NOT_VISUALLY_REVIEWED",
                      limitation="Structured failures screened; image pairs require separate visual review.")
    audit_data = dict(version=VERSION, created_at=datetime.now(timezone.utc).isoformat(),
        scope_dates=scope_dates, scope_basis="User confirmed 20261019 and 20261020 accepted unfolding collection sessions; "
        "session membership, not per-action progress labels; no unified task label in raw logs.",
        accepted_in_scope=len(candidates), usable_transitions=len(rows),
        session_count=len({r["session"] for r in rows}), trajectory_count=len(rows),
        trajectory_definition="One traj_XXXX directory is one recorded executed action; NOT independent sequence.",
        episode_count=len({(r["session"], r["episode_id"]) for r in rows}),
        independent_group_count=len({r["group"] for r in rows}),
        sessions=dict(Counter(r["session"] for r in rows)), action_field_coverage=coverage,
        action_dimensions=len(CONTINUOUS)+1, continuous_fields=CONTINUOUS,
        action_normalization="pixels / (width-1,height-1); dx,dy and distance in normalized image coordinates; "
                             "executed_lift_mm from recorded close-to-lift TCP; all standardized using fit subset only",
        excluded_counts=dict(Counter(r["reason"] for r in excluded)), excluded=excluded,
        candidates=candidates, cross_session_links=links,
        acquisition_review=review,
        limitations=["Assisted/manual records excluded because selected Pa is not actual manual grasp.",
                     "Same garment, viewpoint and collector; held-out sessions do not establish garment generalization.",
                     "No actual grasp/layer/contact sensor labels; remaining acquisition ambiguity retained.",
                     "Known restart sessions and exact-image duplicates are grouped; unlogged interventions are unobservable."],
        robot_calls=0, external_model_calls=0)
    save(output / "data_audit.json", audit_data)
    save(output / "source_hashes.json", source_hashes)
    save(output / "protocol.json", PROTOCOL)
    save(output / "transitions.json", rows)
    write_csv(output / "transitions.csv", rows)
    manifest = dict(split_seed=PROTOCOL["split_seed"], unit="connected whole sessions",
                    assignment="fixed group counts, minimize squared deviation from 20% test/20% validation transition counts; seeded tie-break",
                    groups=splits, transitions={k: [r["id"] for r in rows if r["split"] == k] for k in splits},
                    sessions={k: sorted({r["session"] for r in rows if r["split"] == k}) for k in splits},
                    counts={k: sum(r["split"] == k for r in rows) for k in splits},
                    learning_subsets={}, split_uses_outcomes=False)
    for seed in SEEDS:
        order = list(splits["train"])
        np.random.default_rng(seed).shuffle(order)
        manifest["learning_subsets"][str(seed)] = {}
        for fraction in FRACTIONS:
            selected = order[:max(1, math.ceil(len(order) * fraction))]
            ids = [r["id"] for r in rows if r["split"] == "train" and r["group"] in selected]
            manifest["learning_subsets"][str(seed)][str(fraction)] = dict(
                groups=selected, transition_ids=ids, count=len(ids),
                actual_group_fraction=len(selected)/len(order),
                actual_transition_fraction=len(ids)/manifest["counts"]["train"])
    save(output / "split_manifest.json", manifest)
    print(json.dumps({k: audit_data[k] for k in ("accepted_in_scope", "usable_transitions",
        "session_count", "trajectory_count", "independent_group_count", "excluded_counts")}, ensure_ascii=False), flush=True)
    print("Split counts:", manifest["counts"], flush=True)


def verify_sources(output):
    for path, expected in read(output/"source_hashes.json").items():
        if sha(path) != expected:
            raise ValueError("source changed since audit: " + path)


def encode(output, model_path, threads):
    # This is the only stage that imports/constructs a vision encoder.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoImageProcessor, AutoModel
    torch.set_num_threads(threads)
    verify_sources(output)
    rows = read(output/"transitions.json")
    weights = {p.name: sha(p) for p in model_path.iterdir() if p.is_file()}
    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, use_fast=False)
    fingerprint = hashlib.sha256(json.dumps(dict(weights=weights, processor=processor.to_dict(),
        pooling="L2 normalized last_hidden_state[:,0]", dtype="float32"), sort_keys=True).encode()).hexdigest()
    cache = output/"embeddings"/fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    unique = {}
    for row in rows:
        for side in ("before", "after"):
            unique[row[side+"_pixel_sha256"]] = row[side+"_rgb"]
    needed = [(key, path) for key, path in sorted(unique.items()) if not (cache/(key+".npy")).exists()]
    model = None
    if needed:
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().requires_grad_(False)
        if model.config.model_type != "dinov2" or model.config.hidden_size != 384:
            raise ValueError("frozen DINOv2-small (384) required")
    for i, (key, path) in enumerate(needed):
        with Image.open(path) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            vec = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0].float(), dim=-1)[0].numpy()
        np.save(cache/(key+".npy"), vec)
        if (i+1) % 20 == 0 or i+1 == len(needed):
            print(f"Encoded unique RGB {i+1}/{len(needed)}", flush=True)
    z, y = [np.stack([np.load(cache/(r[side+"_pixel_sha256"]+".npy")) for r in rows])
            for side in ("before", "after")]
    if z.shape[1] != 384 or not np.allclose(np.linalg.norm(np.concatenate([z, y]), axis=1), 1, atol=1e-5):
        raise ValueError("invalid normalized CLS cache")
    np.savez(output/"latents.npz", z=z, y=y, ids=np.array([r["id"] for r in rows]))
    save(output/"encoder_manifest.json", dict(model="facebook/dinov2-small", local_path=str(model_path),
        fingerprint=fingerprint, weights=weights, processor=processor.to_dict(), device="cpu",
        dtype="float32", torch=torch.__version__, pooling="last_hidden_state[:,0], L2 normalized",
        unique_images=len(unique), new_forward_calls=len(needed), reused_images=len(unique)-len(needed),
        latents_sha256=sha(output/"latents.npz"), frozen=True))


def standardize_actions(rows, fit_indices):
    raw = np.array([[r[k] for k in CONTINUOUS] for r in rows], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("missing/nonfinite actions must never be zero-imputed")
    mean, std = raw[fit_indices].mean(0), raw[fit_indices].std(0)
    scale = np.where(std > 1e-8, std, 1.)
    # Zero after centering a genuinely observed constant is not missing-data imputation.
    action = np.column_stack([(raw-mean)/scale, np.ones(len(rows), dtype=np.float32)])
    return action.astype(np.float32), dict(fields=CONTINUOUS, mean=mean.tolist(), scale=scale.tolist(),
        constant_train_fields=[k for k, s in zip(CONTINUOUS, std) if s <= 1e-8],
        categorical_vocabulary=["automatic_pick_place"],
        fit_ids=[rows[i]["id"] for i in fit_indices])


def make_model(action_dim, seed):
    import torch
    torch.manual_seed(seed)
    layers = torch.nn.Sequential(torch.nn.Linear(384+action_dim, 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 384))
    # Residual learning starts exactly at the Identity baseline.
    torch.nn.init.zeros_(layers[-1].weight)
    torch.nn.init.zeros_(layers[-1].bias)
    return layers


def predict(model, z, action=None):
    import torch
    inputs = z if action is None else torch.cat([z, action], dim=1)
    return torch.nn.functional.normalize(z + model(inputs), dim=1)


def group_mean(values, groups):
    return float(np.mean([np.mean(values[np.array(groups) == g]) for g in sorted(set(groups))]))


def train_model(z, y, action, fit, validation, groups, seed, protocol):
    import torch
    model = make_model(0 if action is None else action.shape[1], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    best, best_epoch, stale = -float("inf"), 0, 0
    trace = []
    for epoch in range(protocol["max_epochs"]+1):
        model.eval()
        with torch.no_grad():
            pv = predict(model, z[validation], None if action is None else action[validation])
            val = group_mean((pv*y[validation]).sum(1).numpy(), [groups[i] for i in validation])
        trace.append(dict(epoch=epoch, validation_group_cosine=val))
        if val > best + protocol["min_delta"]:
            best, best_epoch, stale = val, epoch, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= protocol["patience"] or epoch == protocol["max_epochs"]:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = predict(model, z[fit], None if action is None else action[fit])
        loss = (1-(pred*y[fit]).sum(1)).mean()
        loss.backward()
        optimizer.step()
    model.load_state_dict(state)
    model.eval()
    return model, dict(best_epoch=best_epoch, epochs_run=epoch, validation_cosine=best,
        parameters=sum(p.numel() for p in model.parameters()), trace=trace)


def shuffled_donors(count, repeats, seed):
    if count < 2:
        raise ValueError("shuffle needs at least two held-out transitions")
    rng = np.random.default_rng(seed)
    # Uniform draw from all OTHER held-out transitions, with replacement across rows.
    donor = rng.integers(0, count-1, size=(repeats, count))
    return donor + (donor >= np.arange(count)[None, :])


def evaluate(pred, target, identity, groups):
    cosine = (pred*target).sum(1)
    l2 = np.linalg.norm(pred-target, axis=1)
    base_cos = (identity*target).sum(1)
    base_l2 = np.linalg.norm(identity-target, axis=1)
    return dict(cosine_similarity=float(cosine.mean()), latent_l2_error=float(l2.mean()),
        cosine_improvement_over_identity=float((cosine-base_cos).mean()),
        l2_reduction_over_identity=float((base_l2-l2).mean()),
        group_macro_cosine=group_mean(cosine, groups), group_macro_l2=group_mean(l2, groups),
        group_macro_cosine_improvement_over_identity=group_mean(cosine-base_cos, groups))


def train(output, threads):
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    protocol = read(output/"protocol.json")
    rows, split = read(output/"transitions.json"), read(output/"split_manifest.json")
    encoder = read(output/"encoder_manifest.json")
    if sha(output/"latents.npz") != encoder["latents_sha256"]:
        raise ValueError("latent cache hash mismatch")
    data = np.load(output/"latents.npz")
    if data["ids"].tolist() != [r["id"] for r in rows]:
        raise ValueError("latent row order mismatch")
    z, y = torch.from_numpy(data["z"]), torch.from_numpy(data["y"])
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    validation = [i for i, r in enumerate(rows) if r["split"] == "validation"]
    groups = [r["group"] for r in rows]
    tg = [groups[i] for i in test]
    metrics, curve, predictions, diagnostics = [], [], [], []
    (output/"runs").mkdir(exist_ok=True)
    for seed in protocol["seeds"]:
        donors = shuffled_donors(len(test), protocol["shuffle_repeats"], seed+10000)
        np.save(output/f"runs/shuffle_donors_seed{seed}.npy", donors)
        for fraction in protocol["fractions"]:
            subset = split["learning_subsets"][str(seed)][str(fraction)]
            fit = [i for i, r in enumerate(rows) if r["id"] in set(subset["transition_ids"])]
            actions, stats = standardize_actions(rows, fit)
            action = torch.from_numpy(actions)
            run_name = f"seed{seed}_p{int(fraction*100):03d}"
            save(output/f"runs/{run_name}_action_stats.json", stats)
            for name in ("Identity", "State-only", "Action-conditioned"):
                info = dict(best_epoch=0, epochs_run=0, validation_cosine=None, parameters=0)
                if name == "Identity":
                    pred = z[test].numpy()
                else:
                    model, info = train_model(z, y, action if name == "Action-conditioned" else None,
                        fit, validation, groups, seed, protocol)
                    with torch.no_grad():
                        pred = predict(model, z[test], action[test] if name == "Action-conditioned" else None).numpy()
                    torch.save(model.state_dict(), output/f"runs/{run_name}_{name}.pt")
                    save(output/f"runs/{run_name}_{name}_training.json", info)
                record = dict(seed=seed, fraction=fraction, model=name,
                    training_transition_count=len(fit), training_group_count=len(subset["groups"]),
                    actual_training_transition_fraction=subset["actual_transition_fraction"],
                    validation_transition_count=len(validation), heldout_transition_count=len(test),
                    heldout_group_count=len(set(tg)), best_epoch=info["best_epoch"],
                    epochs_run=info["epochs_run"], parameters=info["parameters"],
                    **evaluate(pred, y[test].numpy(), z[test].numpy(), tg))
                curve.append(record)
                cos = (pred*y[test].numpy()).sum(1)
                for j, index in enumerate(test):
                    predictions.append(dict(seed=seed, fraction=fraction, model=name, id=rows[index]["id"],
                        session=rows[index]["session"], group=tg[j], cosine=float(cos[j]),
                        l2=float(np.linalg.norm(pred[j]-y[index].numpy()))))
                if fraction == 1.:
                    metrics.append(record)
                    if name == "Action-conditioned":
                        shuffled_cos, shuffled_l2 = [], []
                        with torch.no_grad():
                            for donor in donors:
                                sp = predict(model, z[test], action[test][donor])
                                shuffled_cos.append((sp*y[test]).sum(1).numpy())
                                shuffled_l2.append(torch.linalg.vector_norm(sp-y[test], dim=1).numpy())
                        sc, sl = np.array(shuffled_cos), np.array(shuffled_l2)
                        for j, index in enumerate(test):
                            diagnostics.append(dict(seed=seed, id=rows[index]["id"], group=tg[j],
                                real_cosine=float(cos[j]), shuffled_cosine=float(sc[:, j].mean()),
                                real_minus_shuffled=float(cos[j]-sc[:, j].mean()),
                                real_beats_shuffle_fraction=float((cos[j] > sc[:, j]).mean()),
                                real_l2=float(np.linalg.norm(pred[j]-y[index].numpy())),
                                shuffled_l2=float(sl[:, j].mean()),
                                repeats=protocol["shuffle_repeats"]))
                print(f"{run_name} {name}: n={len(fit)} cos={record['cosine_similarity']:.6f} "
                      f"group={record['group_macro_cosine']:.6f} epoch={info['best_epoch']}", flush=True)
            write_csv(output/"learning_curve.csv", curve)
            if metrics:
                write_csv(output/"metrics.csv", metrics)
            write_csv(output/"heldout_predictions.csv", predictions)
            if diagnostics:
                write_csv(output/"action_sensitivity.csv", diagnostics)
    save(output/"training_manifest.json", dict(protocol_sha256=sha(output/"protocol.json"),
        split_sha256=sha(output/"split_manifest.json"), latents_sha256=encoder["latents_sha256"],
        module_sha256=sha(__file__), torch=torch.__version__, numpy=np.__version__, device="cpu",
        threads=threads, deterministic_algorithms=True, dino_training_calls=0,
        test_used_for_early_stopping=False, external_model_calls=0, robot_calls=0))


def load_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def paired_evidence(matrix):
    """matrix = seeds x independent groups; bootstrap groups, never adjacent steps."""
    values = np.asarray(matrix, dtype=float)
    means = values.mean(0)
    rng = np.random.default_rng(913)
    draws = means[rng.integers(0, len(means), (PROTOCOL["bootstrap_repeats"], len(means)))].mean(1)
    low, high = np.quantile(draws, [.025, .975])
    return dict(mean=float(values.mean()), seed_positive=int((values.mean(1) > 0).sum()),
                group_positive=int((means > 0).sum()), group_count=len(means),
                group_bootstrap_ci95=[float(low), float(high)])


def report(output):
    # Matplotlib is intentionally needed only for reporting; training never imports it.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/world_model_probe_v1_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    audit_data, split = read(output/"data_audit.json"), read(output/"split_manifest.json")
    metrics = load_csv(output/"metrics.csv")
    curve = load_csv(output/"learning_curve.csv")
    predictions = load_csv(output/"heldout_predictions.csv")
    sensitivity = load_csv(output/"action_sensitivity.csv")
    test_groups = sorted(split["groups"]["test"])
    names = ["Identity", "State-only", "Action-conditioned"]

    def values(name, fraction=1., field="cosine"):
        return np.array([[np.mean([float(r[field]) for r in predictions if r["model"] == name
            and int(r["seed"]) == seed and float(r["fraction"]) == fraction and r["group"] == group])
            for group in test_groups] for seed in SEEDS])

    a, b, c = [values(name) for name in names]
    cb, ca = paired_evidence(c-b), paired_evidence(c-a)
    shuffle = np.array([[np.mean([float(r["real_minus_shuffled"]) for r in sensitivity
        if int(r["seed"]) == seed and r["group"] == group]) for group in test_groups] for seed in SEEDS])
    sh = paired_evidence(shuffle)
    lc = paired_evidence(c-values("Action-conditioned", .8))

    def robust(evidence):
        return (evidence["mean"] >= .001 and evidence["seed_positive"] >= 4
                and evidence["group_bootstrap_ci95"][0] > 0)

    stable = robust(cb) and cb["group_positive"] >= math.ceil(2*len(test_groups)/3)
    beats_identity, uses_action, improving = robust(ca), robust(sh), robust(lc)
    label = "LEARNABLE_SIGNAL" if stable and beats_identity and uses_action else "DATA_LIMITED" if improving else "UNRESOLVED"
    aggregate = {}
    for name in names:
        selected = [r for r in metrics if r["model"] == name]
        aggregate[name] = {}
        for field in ("cosine_similarity", "latent_l2_error", "cosine_improvement_over_identity",
                      "group_macro_cosine", "group_macro_l2"):
            numbers = [float(r[field]) for r in selected]
            aggregate[name][field] = dict(mean=float(np.mean(numbers)), std=float(np.std(numbers, ddof=1)))
    result = dict(classification=label, stable_action_over_state=stable, robust_action_over_identity=beats_identity,
        shuffled_action_clearly_worse=uses_action, learning_curve_still_improving=improving,
        action_minus_state=cb, action_minus_identity=ca, real_minus_shuffled=sh,
        curve_100_minus_80=lc, metrics=aggregate,
        inference_unit="whole connected session groups, paired over the same test transitions",
        caveat="Bootstrap averaged training seeds then resampled held-out groups. Few groups give fragile intervals; "
               "five seeds are not five independent datasets. No causal/counterfactual interpretation.")
    save(output/"conclusions.json", result)
    colors = ["#7b8794", "#3274a1", "#25956a"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, field, title in zip(axes, ["cosine_similarity", "latent_l2_error"],
                                ["Held-out cosine (higher better)", "Latent L2 (lower better)"]):
        means = [aggregate[n][field]["mean"] for n in names]
        stds = [aggregate[n][field]["std"] for n in names]
        ax.bar(names, means, yerr=stds, color=colors, capsize=5)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Transition mean; error bars = seed SD")
    fig.tight_layout()
    fig.savefig(output/"baseline_comparison.png", dpi=170)
    fig.savefig(output/"baseline_comparison.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, color in zip(names, colors):
        xx, yy, ee = [], [], []
        for fraction in FRACTIONS:
            part = [r for r in curve if r["model"] == name and float(r["fraction"]) == fraction]
            xx.append(np.mean([int(r["training_transition_count"]) for r in part]))
            yy.append(np.mean([float(r["group_macro_cosine"]) for r in part]))
            ee.append(np.std([float(r["group_macro_cosine"]) for r in part], ddof=1))
            if name == "Action-conditioned":
                ax.annotate(f"{fraction:.0%}", (xx[-1], yy[-1]), xytext=(0, 8), textcoords="offset points", fontsize=8)
        ax.errorbar(xx, yy, yerr=ee, marker="o", capsize=4, color=color, label=name)
    ax.set(xlabel="Training transitions (mean across 5 nested group orders)",
           ylabel="Fixed held-out group-macro cosine", title="Learning curve; bars = seed SD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"learning_curve.png", dpi=170)
    fig.savefig(output/"learning_curve.svg")
    plt.close(fig)
    real = c.mean(1)
    shuffled = (c-shuffle).mean(1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (r, s) in enumerate(zip(real, shuffled)):
        ax.plot([0, 1], [r, s], marker="o", label=f"seed {SEEDS[i]}")
    ax.set(xticks=[0, 1], xticklabels=["Real action", "Shuffled action"],
           ylabel="Held-out group-macro cosine", title="Action usage diagnostic (100 shuffles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"real_vs_shuffled.png", dpi=170)
    fig.savefig(output/"real_vs_shuffled.svg")
    plt.close(fig)
    lines = [
        "# world_model_probe_v1", "", f"当前证据分类：**{label}**。本轮仅诊断，不自动换模型、改表示或加数据。", "",
        f"- 最终 usable transitions：**{audit_data['usable_transitions']}**；"
        f"**{audit_data['session_count']} sessions**、**{audit_data['trajectory_count']} 单动作 traj 记录**、"
        f"**{audit_data['independent_group_count']} 独立分组**。",
        f"- Action-conditioned 是否稳定优于 State-only：**{'是' if stable else '否，未满足预先固定的稳定性标准'}**。"
        f"组均值 cosine 差 {cb['mean']:+.6f}；{cb['seed_positive']}/5 seeds 为正，"
        f"{cb['group_positive']}/{cb['group_count']} test groups 为正；95% group bootstrap CI {cb['group_bootstrap_ci95']}。",
        f"- Action-conditioned 是否优于 Identity：**{'是（稳健）' if beats_identity else '未获得稳健改善'}**。"
        f"组均值差 {ca['mean']:+.6f}；{ca['seed_positive']}/5 seeds 为正；CI {ca['group_bootstrap_ci95']}。",
        f"- Shuffled action 是否明显更差：**{'是' if uses_action else '否，未达到明显差异标准'}**。"
        f"真实 action − shuffled action = {sh['mean']:+.6f}；{sh['seed_positive']}/5 seeds 为正；"
        f"CI {sh['group_bootstrap_ci95']}。",
        f"- Learning curve 到 100% 是否仍明显改善：**{'是' if improving else '未证实'}**。"
        f"100% − 80% = {lc['mean']:+.6f}；{lc['seed_positive']}/5 seeds 为正；CI {lc['group_bootstrap_ci95']}。",
        "", "| Model | Test cosine mean ± seed SD | Latent L2 mean ± seed SD | Δ cosine vs Identity | Group-macro cosine |",
        "|---|---:|---:|---:|---:|"]
    for name in names:
        m = aggregate[name]
        lines.append(f"| {name} | {m['cosine_similarity']['mean']:.6f} ± {m['cosine_similarity']['std']:.6f} | "
            f"{m['latent_l2_error']['mean']:.6f} ± {m['latent_l2_error']['std']:.6f} | "
            f"{m['cosine_improvement_over_identity']['mean']:+.6f} | {m['group_macro_cosine']['mean']:.6f} |")
    lines += ["", "![Baseline comparison](baseline_comparison.png)", "",
        f"划分：train={split['counts']['train']}、validation={split['counts']['validation']}、test={split['counts']['test']} transitions；"
        f"分组数分别为 {len(split['groups']['train'])}/{len(split['groups']['validation'])}/{len(split['groups']['test'])}。"
        "按整 session 及已记录跨 session 连续性合并，不随机拆相邻 step。分组和 seed 在训练前冻结。"
        "validation 只用于 early stopping；test 不用于标准化、选 epoch、调参或改模型。", "",
        "DINOv2-small frozen，完整 RGB 使用既有 processor（其默认 resize/center crop 保持不变），"
        "取 384 维 CLS 后 L2 normalize，按像素 SHA256 去重离线缓存；MLP 训练仅加载 latents.npz。"
        "像素归一化 Pa/Pb 与执行 motion 的 XY 一致性已核验；dx/dy/distance 从记录坐标算出，"
        "lift 来自实际执行脚本。8 个连续字段及 1 维 primitive one-hot。每个 learning subset 独立拟合均值/标准差，"
        "不使用 validation/test 统计量。唯一 primitive 为 automatic_pick_place，因此本轮不能辨识 primitive 类型差异。", "",
        "B/C 均为 Linear(256) → GELU → Linear(256) → GELU → Linear(384)，输出残差后 normalize；"
        "最后一层零初始化使 epoch 0 等于 Identity。C 仅增加 action 输入对应权重，无额外隐藏层。"
        "AdamW lr=0.001、weight_decay=0.01，最多 600 epochs，patience=60；5 个固定 seeds，所有比例相同配置。", "",
        "| Requested train group fraction | Training transitions across seeds | Test group-macro C cosine mean ± SD |",
        "|---|---|---:|"]
    for fraction in FRACTIONS:
        part = [r for r in curve if r["model"] == "Action-conditioned" and float(r["fraction"]) == fraction]
        v = [float(r["group_macro_cosine"]) for r in part]
        lines.append(f"| {fraction:.0%} | {', '.join(r['training_transition_count'] for r in part)} | "
                     f"{np.mean(v):.6f} ± {np.std(v, ddof=1):.6f} |")
    lines += ["", "![Learning curve](learning_curve.png)", "",
        "20/40/60/80/100% 指 train groups 的比例，向上取整；不是随机抽同样比例的 steps。"
        "每个 seed 使用一个嵌套 group 顺序，B/C 同 seed 同子集，同一 test 固定不变。"
        "learning_curve.csv 同时记录真实 transition 数量及比例。", "",
        "![Action sensitivity](real_vs_shuffled.png)", "",
        "每个 test transition 从其他 test transitions 均匀抽 action（不抽自己），重复 100 次，"
        "只替换 action、保持 z_t 和目标不变。该结果仅检测 action 使用，**不是物理 counterfactual dynamics**。", "",
        f"数据审计：scope 内 accepted={audit_data['accepted_in_scope']}。排除计数："
        f"`{json.dumps(audit_data['excluded_counts'], ensure_ascii=False)}`。"
        "仅用专门拍摄的 after；不桥接 reset/pending，不把下一步 before 冒充 after。"
        f"{audit_data['excluded_counts'].get('assisted_actual_grasp_not_representable_by_selected_Pa', 0)} "
        "条 assisted 手柄动作的选点不能作为真实抓点，本轮未猜测转换；逐条原因在 data_audit.json。"
        f"视觉审阅覆盖 {audit_data['acquisition_review']['candidate_pairs_reviewed']} 对 before/after；"
        "具体发现见 data_audit.json 的 acquisition_review；不能据此证明所有动作 acquisition 成功。", "",
        "稳定性标准：C−B 在 ≥4/5 seeds、≥2/3 test groups 为正，平均至少 0.001，"
        "且以独立 group 为单位的配对 bootstrap 95% CI 下界 >0。Identity/shuffle/curve 使用同样幅度与 seed/CI 标准。"
        "Bootstrap 先对 seeds 平均，再抽 test groups；5 seeds 并非 5 份独立数据。", "",
        "**解释边界：** 少量 held-out groups、单件衣物和同一相机限制结论。"
        "无论无提升还是曲线平台，都不能单凭本实验判定 REPRESENTATION_LIMITED；还可能是数据量、"
        "动作覆盖、隐藏布层/抓取歧义或训练估计方差。DATA_LIMITED 仅在预定曲线标准成立时采用，"
        "否则保守归为 UNRESOLVED。未满足 LEARNABLE_SIGNAL 条件时，不据此进入下一步 world model。", "",
        "完整可追溯文件：data_audit.json、transitions.csv、split_manifest.json、protocol.json、"
        "encoder_manifest.json、metrics.csv、learning_curve.csv、action_sensitivity.csv、"
        "heldout_predictions.csv、conclusions.json；runs/ 保存每次 action 标准化统计、checkpoint、epoch 记录与 shuffle donor 索引。", ""]
    (output/"summary.md").write_text("\n".join(lines))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "encode", "train", "report"])
    parser.add_argument("--output", type=Path, default=Path("reports/world_model_probe_v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/demonstrations"))
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--scope-date", action="append")
    args = parser.parse_args()
    if args.stage == "audit":
        audit(args.data_root, args.output, args.scope_date or ["20261019", "20261020"])
    elif args.stage == "encode":
        encode(args.output, args.model_path, args.threads)
    elif args.stage == "train":
        train(args.output, args.threads)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
