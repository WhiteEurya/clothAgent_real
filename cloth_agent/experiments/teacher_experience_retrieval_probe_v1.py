"""Offline teacher-citation retrieval. No model service, graph, or robot imports."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

# Set NumPy BLAS threads before NumPy is imported; no optional runtime dependency.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAME = "teacher_experience_retrieval_probe_v1"
EPISODE = Path("reports/trajectory_memory_full_episode_v1_collection_aligned")
OUTPUT = Path("reports")/NAME
MODEL = Path("/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small")
BASELINE = Path("cloth_agent/experiments/unfold_visual_retrieval_benchmark.py")
OLD_CACHE = Path("reports/unfold_visual_retrieval_benchmark_20261019/memory/memory.json")
KS = (1, 3, 5, 10, 20)
STAGES = ("decode_ms", "preprocess_ms", "forward_ms", "exact_ranking_ms",
          "ann_search_ms", "total_exact_ms", "total_with_ann_validation_ms")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_sha(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()


def csv_save(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_components(path=BASELINE):
    """Reuse exact existing pure-vision definitions without importing GPT/Graph code."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    names = {"decode", "DinoEncoder", "build_index"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("existing visual pipeline definitions missing")
    return "\n\n".join(ast.get_source_segment(text, n) for n in nodes)+"\n"


def manifest_inputs(episode):
    episode = Path(episode).resolve()
    frozen = read(episode/"memory_manifest.json")
    refs = []
    for record in frozen["records"]:
        tid, sid = record["trajectory_id"], record["step_id"]
        relative = f"memory/{tid}/{sid}/before.png"
        path = episode/relative
        if relative not in frozen["files"] or sha(path) != frozen["files"][relative]:
            raise ValueError("reference absent from or inconsistent with frozen memory: "+relative)
        refs.append(dict(experience_id=f"{tid}/{sid}", trajectory_id=tid, step_id=sid,
                         trajectory_unit=tid, before_rgb_path=str(path), sha256=sha(path),
                         pixel_sha256=pixel_sha(path)))
    refs.sort(key=lambda r: r["experience_id"])
    ids = {r["experience_id"] for r in refs}
    if len(ids) != len(refs):
        raise ValueError("duplicate experience identifiers")
    queries, excluded = [], []
    for folder in sorted(episode.glob("round_*")):
        if not (folder/"plan.json").is_file():
            excluded.append(dict(round=folder.name, reason="no saved plan.json"))
            continue
        req = folder/"agent_events/request.json"
        metrics = folder/"agent_events/metrics.json"
        if not req.is_file() or not metrics.is_file():
            raise ValueError("plan lacks evidence of actual Agent call")
        plan = read(folder/"plan.json")
        positives = []
        for citation in plan["supporting_experiences"]:
            eid = f"{citation['trajectory_id']}/{citation['step_id']}"
            if eid not in ids:
                raise ValueError("teacher citation outside frozen gallery: "+eid)
            positives.append(dict(experience_id=eid, trajectory_id=citation["trajectory_id"],
                                  step_id=citation["step_id"], use=citation["use"]))
        if not positives:
            raise ValueError("query has no teacher citations; cannot silently drop it")
        image = folder/"before/rgb.png"
        if sha(image) != read(req)["current_observation_sha256"]:
            raise ValueError("query differs from actual Agent input")
        if sha(image) != read(folder/"before/observation.json")["sha256"]["rgb.png"]:
            raise ValueError("query observation provenance mismatch")
        queries.append(dict(query_round_id=folder.name, query_rgb_path=str(image),
                            query_sha256=sha(image), pixel_sha256=pixel_sha(image),
                            plan_path=str(folder/"plan.json"), plan_sha256=sha(folder/"plan.json"),
                            decision=plan["decision"], execution=read(folder/"execution.json")["status"],
                            positives=positives, teacher_positive_ids=sorted({p["experience_id"] for p in positives})))
    return frozen, refs, queries, excluded


def verify_inputs(out):
    for path, expected in read(out/"input_hashes.json").items():
        if sha(path) != expected:
            raise ValueError("input changed: "+path)


def audit(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    frozen, refs, queries, excluded = manifest_inputs(args.episode)
    if len(refs) != 139 or len(queries) != 7:
        raise ValueError(f"unexpected scope: {len(refs)} references, {len(queries)} queries")
    episode = args.episode.resolve()
    # Hash-only provenance verification of all frozen files; their contents never enter retrieval.
    hashes = {str(episode/name): value for name, value in frozen["files"].items()}
    hashes[str(episode/"memory_manifest.json")] = sha(episode/"memory_manifest.json")
    for q in queries:
        hashes[q["query_rgb_path"]] = q["query_sha256"]
        hashes[q["plan_path"]] = q["plan_sha256"]
    hashes.update({str(p.resolve()): sha(p) for p in args.model_path.iterdir() if p.is_file()})
    hashes[str(BASELINE.resolve())] = sha(BASELINE)
    save(out/"input_hashes.json", hashes)
    verify_inputs(out)
    previous = read(OLD_CACHE)
    old_pixels = {r["pixel_sha256"] for r in previous["images"]}
    coverage = sum(r["pixel_sha256"] in old_pixels for r in refs)
    save(out/"teacher_positive_manifest.json", dict(
        source_episode=str(episode), queries=queries, excluded_rounds=excluded,
        query_count=len(queries), citation_occurrences=sum(len(q["positives"]) for q in queries),
        distinct_cited_steps=len({p for q in queries for p in q["teacher_positive_ids"]}),
        labels="Saved teacher citations only; not exhaustive relevance or optimal-action ground truth",
        reason_text_usage="audit/report only; never encoder or ranking input"))
    save(out/"reference_index_manifest.json", dict(
        source_frozen_manifest=str(episode/"memory_manifest.json"), frozen_at=frozen["frozen_at"],
        reference_count=len(refs), trajectory_unit_count=len({r["trajectory_unit"] for r in refs}),
        trajectory_unit_definition="trajectory_id = historical collection session; step_id is an experience",
        references=refs, encoder_input="historical BEFORE RGB only",
        cache_audit=dict(previous_manifest=str(OLD_CACHE.resolve()), previous_rows=len(previous["images"]),
                         covered_before_rows=coverage, exact_gallery_match=False,
                         decision="precompute all 139 before images once; prior gallery is 120 mixed before/after observations"),
        query_pixel_duplicates_in_gallery={q["query_round_id"]:
            [r["experience_id"] for r in refs if r["pixel_sha256"] == q["pixel_sha256"]] for q in queries}))
    config = dict(experiment=NAME, fixed_before_ranking=True, created_at=datetime.now(timezone.utc).isoformat(),
        model_path=str(args.model_path.resolve()), device=args.device, ks=list(KS),
        repeats_per_query=20, warmup_per_query=2, latency_schedule_seed=20261020,
        reference_encoding="once before runtime", accuracy_query_encoding="once; latency repeats are separate",
        preprocessing="Exact prior DinoEncoder: local AutoImageProcessor use_fast=False; unchanged model config",
        pooling="CLS 384d float32 L2 normalized", crop_tuning=False, input="RGB only",
        rank="exact normalized inner product descending; ties retain experience_id order",
        ann=dict(type="FAISS HNSW inner product", M=32, efConstruction=200, efSearch=64, threads=1),
        diversity="retain highest exact-scoring step per trajectory_id, original teacher positives unchanged",
        classification_rule=dict(
            UNRESOLVED="incomplete audit or fewer than 5 valid queries",
            STRONG_RETRIEVAL_SIGNAL="Hit@5 >= 0.70 AND macro CitationRecall@5 >= 0.70",
            MODERATE_RETRIEVAL_SIGNAL="not strong, Hit@20 >= 0.70 AND macro CitationRecall@20 >= 0.70",
            WEAK_RETRIEVAL_SIGNAL="complete audit with >=5 queries, neither strong nor moderate",
            scope="Descriptive labels for this seven-query episode only, not statistical generalization"),
        torch_threads=24, numpy_threads=1, code_sha256=sha(__file__), baseline_source_sha256=sha(BASELINE),
        forbidden=["GPT/Codex calls", "training", "robot", "Graph", "feature/K/crop tuning"])
    save(out/"protocol.json", config)
    (out/"reused_visual_pipeline.py.txt").write_text(source_components())
    shutil.copyfile(__file__, out/"experiment_source.py")
    print(json.dumps(dict(stage="AUDITED", queries=len(queries), references=len(refs),
                          citations=sum(len(q["positives"]) for q in queries), old_cache_covered=coverage)), flush=True)


def exact_order(vectors, query):
    scores = vectors @ query.reshape(-1)
    return np.argsort(-scores, kind="stable"), scores


def diverse_order(order, refs):
    seen, chosen = set(), []
    for i in order:
        group = refs[int(i)]["trajectory_unit"]
        if group not in seen:
            seen.add(group)
            chosen.append(int(i))
    return np.asarray(chosen, dtype=int)


def query_metrics(order, refs, positives):
    ids = [refs[int(i)]["experience_id"] for i in order]
    ranks = [i+1 for i, eid in enumerate(ids) if eid in positives]
    result = dict(best_teacher_rank=min(ranks) if ranks else None,
                  reciprocal_rank=1/min(ranks) if ranks else 0., retained_teacher_count=len(ranks),
                  teacher_positive_count=len(positives), candidate_count=len(ids))
    for k in KS:
        hits = len(set(ids[:k]) & set(positives))
        result[f"hit_at_{k}"] = int(hits > 0)
        result[f"citation_recall_at_{k}"] = hits/len(positives)
    return result


def aggregate(rows):
    ranks = [r["best_teacher_rank"] for r in rows]
    finite = [r for r in ranks if r is not None]
    result = dict(query_count=len(rows), mrr=float(np.mean([r["reciprocal_rank"] for r in rows])),
                  missing_teacher_queries=len(rows)-len(finite),
                  best_teacher_rank=dict(
                      median=float(np.median(finite)) if len(finite) == len(rows) else None,
                      mean=float(np.mean(finite)) if len(finite) == len(rows) else None,
                      max=max(finite) if len(finite) == len(rows) else None,
                      missing_rank_semantics="infinite; JSON null if any query has no retained cited step",
                      per_query={r["query_round_id"]: r["best_teacher_rank"] for r in rows}))
    for metric in ("hit", "citation_recall"):
        result[metric] = {str(k): float(np.mean([r[f"{metric}_at_{k}"] for r in rows])) for k in KS}
    return result


def classification(primary, complete=True):
    if not complete or primary["query_count"] < 5:
        return "UNRESOLVED"
    if primary["hit"]["5"] >= .7 and primary["citation_recall"]["5"] >= .7:
        return "STRONG_RETRIEVAL_SIGNAL"
    if primary["hit"]["20"] >= .7 and primary["citation_recall"]["20"] >= .7:
        return "MODERATE_RETRIEVAL_SIGNAL"
    return "WEAK_RETRIEVAL_SIGNAL"


def stats(values):
    return dict(median=float(np.median(values)), mean=float(np.mean(values)),
                p95=float(np.percentile(values, 95)), count=len(values))


def contact_sheet(out, query, order, scores, refs):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    canvas = Image.new("RGB", (1520, 1200), "white")
    draw = ImageDraw.Draw(canvas)
    tiles = [(query["query_rgb_path"], ["QUERY "+query["query_round_id"], "Current before RGB"], False)]
    for rank, i in enumerate(order[:20], 1):
        ref = refs[int(i)]
        cited = ref["experience_id"] in query["teacher_positive_ids"]
        tiles.append((ref["before_rgb_path"], [f"#{rank}  cosine={scores[i]:.5f}",
            ref["trajectory_id"], ref["step_id"]+("  TEACHER_CITED" if cited else "")], cited))
    for n, (path, labels, cited) in enumerate(tiles):
        x, y = (n % 5)*304, (n//5)*240
        draw.rectangle((x, y, x+301, y+237), outline="#168238" if cited else "#aaa", width=4 if cited else 1)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((296, 167))
            canvas.paste(image, (x+4, y+4))
        for j, label in enumerate(labels):
            draw.text((x+6, y+175+j*19), label, font=font, fill="#126b2f" if cited else "black")
    folder = out/"contact_sheets"
    folder.mkdir(exist_ok=True)
    canvas.save(folder/(query["query_round_id"]+"_top20.png"))


def run(args):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      MPLCONFIGDIR="/tmp/teacher_retrieval_mpl")
    out = args.output.resolve()
    config = read(out/"protocol.json")
    if config["code_sha256"] != sha(__file__):
        raise ValueError("implementation changed after fixed audit")
    if (out/"summary.json").exists() or (out/"reference_embeddings.npy").exists():
        raise ValueError("refuse repeated/tuned run in the same output")
    verify_inputs(out)
    import torch
    import faiss
    torch.set_num_threads(config["torch_threads"])
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently change the registered device")
    namespace = dict(np=np, time=time, Path=Path, Image=Image, file_hash=sha)
    exec(compile((out/"reused_visual_pipeline.py.txt").read_text(), "reused_visual_pipeline", "exec"), namespace)
    encoder = namespace["DinoEncoder"](config["model_path"], config["device"])
    if encoder.model.config.hidden_size != 384:
        raise ValueError("expected DINOv2-small CLS dimension 384")
    old_meta = read(OLD_CACHE)["encoder"]
    if encoder.meta["weights"] != old_meta["weights"] or encoder.meta["processor"] != old_meta["processor"]:
        raise ValueError("encoder or preprocessing differs from the verified baseline")
    save(out/"encoder_manifest.json", dict(**encoder.meta, frozen=True, training=False,
                                          reused_definitions=["decode", "DinoEncoder", "build_index"]))
    refs_manifest = read(out/"reference_index_manifest.json")
    refs = refs_manifest["references"]
    queries = read(out/"teacher_positive_manifest.json")["queries"]
    decode = namespace["decode"]
    began = time.perf_counter()
    vectors = []
    for i, ref in enumerate(refs):
        image = decode(ref["before_rgb_path"])
        vector, _, _ = encoder.encode(image)
        image.close()
        vectors.append(vector[0])
        if (i+1) % 25 == 0:
            print(f"REFERENCE ENCODE {i+1}/{len(refs)}", flush=True)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.shape != (139, 384) or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5):
        raise ValueError("invalid reference embeddings")
    np.save(out/"reference_embeddings.npy", vectors)
    index = namespace["build_index"](vectors)
    faiss.write_index(index, str(out/"reference.index"))
    refs_manifest.update(embedding_shape=list(vectors.shape), reference_encode_calls=139,
        precompute_s=time.perf_counter()-began, embeddings_sha256=sha(out/"reference_embeddings.npy"),
        faiss_index_sha256=sha(out/"reference.index"), faiss_version=faiss.__version__)
    save(out/"reference_index_manifest.json", refs_manifest)
    primary, secondary, ranks, div_ranks, agreement, qvectors = [], [], [], [], [], []
    for q in queries:
        image = decode(q["query_rgb_path"])
        vector, _, _ = encoder.encode(image)
        image.close()
        qvectors.append(vector[0])
        order, scores = exact_order(vectors, vector)
        diverse = diverse_order(order, refs)
        common = dict(query_round_id=q["query_round_id"], teacher_cited=";".join(q["teacher_positive_ids"]))
        primary.append(dict(**common, variant="primary", **query_metrics(order, refs, q["teacher_positive_ids"])))
        secondary.append(dict(**common, variant="diversity", **query_metrics(diverse, refs, q["teacher_positive_ids"])))
        for variant, ordering, dest in (("primary", order, ranks), ("diversity", diverse, div_ranks)):
            for rank, i in enumerate(ordering, 1):
                ref = refs[int(i)]
                dest.append(dict(query_round_id=q["query_round_id"], rank=rank, experience_id=ref["experience_id"],
                    trajectory_id=ref["trajectory_id"], step_id=ref["step_id"], cosine_score=float(scores[i]),
                    is_teacher_cited=ref["experience_id"] in q["teacher_positive_ids"]))
        ann_scores, ann_ids = index.search(vector, max(KS))
        agreement.append(dict(query_round_id=q["query_round_id"], checks={
            str(k): dict(same_set=set(order[:k]) == set(ann_ids[0, :k]),
                         same_order=order[:k].tolist() == ann_ids[0, :k].tolist()) for k in KS},
            max_score_error=float(np.max(np.abs(ann_scores[0]-scores[ann_ids[0]]))),
            ann_top20_ids=[refs[int(i)]["experience_id"] for i in ann_ids[0]]))
        contact_sheet(out, q, order, scores, refs)
    np.save(out/"query_embeddings.npy", np.ascontiguousarray(qvectors))
    csv_save(out/"rankings.csv", ranks)
    csv_save(out/"diversity_rankings.csv", div_ranks)
    csv_save(out/"per_query_metrics.csv", primary+secondary)
    save(out/"ann_agreement.json", dict(per_query=agreement,
        all_topk_sets_equal=all(c["same_set"] for r in agreement for c in r["checks"].values()),
        all_topk_orders_equal=all(c["same_order"] for r in agreement for c in r["checks"].values()),
        policy="Exact ranking authoritative. HNSW settings fixed; no retry/tuning if mismatch."))
    # Accuracy uses one query vector each. Re-encoding below is exclusively latency measurement.
    def timed(q):
        encoder.sync()
        start = time.perf_counter()
        image = decode(q["query_rgb_path"])
        decode_end = time.perf_counter()
        vector, prep, forward = encoder.encode(image)
        image.close()
        t = time.perf_counter()
        order, _ = exact_order(vectors, vector)
        ranked = time.perf_counter()
        index.search(vector, max(KS))
        finished = time.perf_counter()
        return dict(query_round_id=q["query_round_id"], decode_ms=(decode_end-start)*1000,
            preprocess_ms=prep*1000, forward_ms=forward*1000, exact_ranking_ms=(ranked-t)*1000,
            ann_search_ms=(finished-ranked)*1000, total_exact_ms=(ranked-start)*1000,
            total_with_ann_validation_ms=(finished-start)*1000)
    print("LATENCY warmup, then 20 repetitions per query; reference embeddings fixed", flush=True)
    for q in queries:
        for _ in range(config["warmup_per_query"]):
            timed(q)
    latency = []
    rng = np.random.default_rng(config["latency_schedule_seed"])
    for repeat in range(config["repeats_per_query"]):
        for qi in rng.permutation(len(queries)):
            latency.append(dict(repeat=repeat, **timed(queries[int(qi)])))
        if (repeat+1) % 5 == 0:
            print(f"LATENCY {repeat+1}/20 completed", flush=True)
    csv_save(out/"latency_samples.csv", latency)
    lat_summary = dict(units="ms", pooled={s: stats([r[s] for r in latency]) for s in STAGES},
        per_query={q["query_round_id"]: {s: stats([r[s] for r in latency if r["query_round_id"] == q["query_round_id"]])
                    for s in STAGES} for q in queries},
        warmup_per_query=config["warmup_per_query"], repeats_per_query=20, samples=len(latency),
        notes=["batch size 1; warm filesystem cache; model/index already resident",
               "forward includes host-to-device transfer, CLS normalization, CPU copy and CUDA synchronization",
               "total_exact is decode + processing + forward + full exact ranking + host overhead",
               "total_with_ann_validation additionally searches fixed HNSW Top-20",
               "stage quantiles are not additive; no reference encoding, sheet rendering, or model startup in runtime"])
    save(out/"latency_summary.json", lat_summary)
    p, s = aggregate(primary), aggregate(secondary)
    verify_inputs(out)
    teacher = read(out/"teacher_positive_manifest.json")
    summary = dict(experiment=NAME, primary=p, diversity=s, query_count=len(queries), reference_count=len(refs),
        citation_occurrences=teacher["citation_occurrences"], distinct_cited_steps=teacher["distinct_cited_steps"],
        judgment=classification(p), classification_scope="fixed descriptive rule; only one seven-query episode",
        latency=lat_summary, ann_agreement=read(out/"ann_agreement.json"),
        input_hashes_unchanged=True, encoder_forward_calls=encoder.calls,
        expected_encoder_forward_calls=139+7+7*config["warmup_per_query"]+7*20,
        reference_reencoded_during_runtime=False, model_api_calls=0, training=False, robot_connections=0,
        exact_rankings_count=len(ranks), diversity_rankings_count=len(div_ranks),
        pipeline_unchanged_from_verified_baseline=True)
    if encoder.calls != summary["expected_encoder_forward_calls"]:
        raise ValueError("unexpected encoder call count")
    save(out/"summary.json", summary)
    render(out, summary, queries, primary, secondary)
    print(json.dumps(dict(stage="COMPLETE", judgment=summary["judgment"], hit=p["hit"],
                          citation_recall=p["citation_recall"], latency=lat_summary["pooled"]["total_exact_ms"])), flush=True)


def render(out, summary, queries, primary, secondary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p, d, lat = summary["primary"], summary["diversity"], summary["latency"]
    x = np.arange(len(queries))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, [r["best_teacher_rank"] for r in primary])
    ax.set_xticks(x, [q["query_round_id"] for q in queries], rotation=25)
    ax.set_ylabel("Best teacher-cited rank (lower is better)")
    for k in (5, 20):
        ax.axhline(k, color="gray", linestyle="--", linewidth=.8)
    fig.tight_layout()
    fig.savefig(out/"best_teacher_rank.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("hit", "citation_recall"), ("Teacher Hit@K", "Macro Citation Recall@K")):
        ax.plot(KS, [p[metric][str(k)] for k in KS], "o-", label="Primary exact")
        ax.plot(KS, [d[metric][str(k)] for k in KS], "s--", label="One per session")
        ax.set(xticks=KS, ylim=(-.03, 1.05), xlabel="K", title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out/"hit_and_citation_recall.png", dpi=160)
    plt.close(fig)
    stages = STAGES[:5]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(5)-.18, [lat["pooled"][s]["median"] for s in stages], .36, label="median")
    ax.bar(np.arange(5)+.18, [lat["pooled"][s]["p95"] for s in stages], .36, label="p95")
    ax.set(xticks=np.arange(5), xticklabels=["decode", "preprocess", "forward", "exact ranking", "ANN"],
           ylabel="Milliseconds (log scale)", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"latency_breakdown.png", dpi=160)
    plt.close(fig)
    pct = lambda value: f"{100*value:.1f}%"
    lines = ["# teacher_experience_retrieval_probe_v1", "",
        f"**本次固定协议判断：{summary['judgment']}。** 这是单个真实 episode 的 teacher-citation 回收诊断；"
        "不是 state matching、Graph node accuracy、动作质量或因果效益评估。", "",
        "## 数据与固定方法", "",
        f"- Query：**{summary['query_count']}** 个实际调用 Agent 且保存 plan 的 round，均为已执行 ACTION；round_007 无 plan，排除。",
        f"- 从保存的 plan 原样解析 **{summary['citation_occurrences']} 次 teacher citations、{summary['distinct_cited_steps']} 个 distinct cited steps**；同轮多引用全部保留。",
        "- 候选：episode 开始前冻结的 **139 个 experience / 16 个 session**；只将每条 historical BEFORE RGB 输入检索器。",
        "- 原基准缓存为 120 张 before/after 混合图，不对应本次候选集；本次 139 张 reference 各编码一次。评估 query 各编码一次；延迟实验另行重复。",
        "- 直接复用旧脚本的 DinoEncoder / decode / build_index 定义；只提取这三个纯视觉定义，不导入 GPT、Graph 或机器人模块。",
        "- 本地 frozen DINOv2-small、CLS 384d、float32、L2 normalize；exact inner product 为 primary。",
        "- 原处理器：RGB → 短边 resize 256（bicubic）→ 中心裁剪 224×224 → rescale/normalize。这里的 whole-image 指没有衣物/局部自定义 crop；保留处理器原有 center crop，不改特征。",
        "- K 固定为 1/3/5/10/20；不删除相邻 step。完整排名保留每个候选，未使用 action/after/depth/文字 rationale。",
        "- 全部源文件哈希前后核验不变；未调用 GPT/Codex、训练、修改历史 memory/Graph 或连接机器人。", "",
        "## Primary 结果", "", "|K|Teacher Hit@K|Macro Citation Recall@K|", "|---:|---:|---:|"]
    for k in KS:
        lines.append(f"|{k}|{pct(p['hit'][str(k)])}|{pct(p['citation_recall'][str(k)])}|")
    lines += ["", f"Best teacher rank：median **{p['best_teacher_rank']['median']:.1f}**，"
              f"mean **{p['best_teacher_rank']['mean']:.2f}**，max **{p['best_teacher_rank']['max']}**；"
              f"MRR **{p['mrr']:.4f}**。", "",
              "|Round|Teacher cited（trajectory/step）|Best cited rank|Hit@3|Hit@5|Hit@10|Top-20 图|",
              "|---|---|---:|---:|---:|---:|---|"]
    for q, r in zip(queries, primary):
        lines.append(f"|{q['query_round_id']}|{'<br>'.join(q['teacher_positive_ids'])}|{r['best_teacher_rank']}|"
                     f"{r['hit_at_3']}|{r['hit_at_5']}|{r['hit_at_10']}|"
                     f"[contact sheet](contact_sheets/{q['query_round_id']}_top20.png)|")
    lines += ["", "![best rank](best_teacher_rank.png)", "", "![Hit and Recall](hit_and_citation_recall.png)", "",
        "## Diversity secondary", "",
        "按 `trajectory_id`（collection session）分组，每组只留下 exact ranking 最高分 step，因此仅 16 个候选。"
        "这不是把同 session 的未引用 step 当成命中；正例仍是 teacher 原来引用的具体 step，Recall 分母也不减少。"
        "被去掉的 cited step 视为无法召回；因此 K=20 不保证 Hit/Recall=100%。", "",
        "|K|Diversity Hit@K|相对 primary Hit 差值（百分点）|Diversity Citation Recall@K|",
        "|---:|---:|---:|---:|"]
    for k in KS:
        lines.append(f"|{k}|{pct(d['hit'][str(k)])}|{100*(d['hit'][str(k)]-p['hit'][str(k)]):+.1f}|"
                     f"{pct(d['citation_recall'][str(k)])}|")
    max_hit_delta = max(abs(d["hit"][str(k)]-p["hit"][str(k)]) for k in KS)
    max_recall_delta = max(abs(d["citation_recall"][str(k)]-p["citation_recall"][str(k)]) for k in KS)
    lines += ["", f"是否明显变化：五档 K 中最大 Hit 变化为 **{100*max_hit_delta:.1f} 个百分点**，"
              f"最大 Recall 变化为 **{100*max_recall_delta:.1f} 个百分点**。"
              + ("至少一项变化达到一个 query 的宏平均权重（1/7≈14.3 个百分点），在这组样本上有明显数值变化。"
                 if max(max_hit_delta, max_recall_delta) >= 1/7-1e-9 else
                 "在这组样本上未达到一个 query 的宏平均权重（1/7≈14.3 个百分点）。")
              + "此处只描述幅度，不表示统计显著性。"]
    lines += ["", f"Diversity MRR={d['mrr']:.4f}；没有保留任何 cited step 的 query 数="
              f"{d['missing_teacher_queries']}。完整 secondary 的 best rank / mean / median / max 与逐轮指标见 summary.json 和 per_query_metrics.csv；"
              "缺失 rank 记为无限，JSON 用 null 表示，MRR 贡献为 0，不剔除困难 query。",
              "", "## Exact 与 ANN", "",
              f"FAISS HNSW M=32、efConstruction=200、efSearch=64、1 thread，参数沿用原流程。"
              f"所有 query 的五档 Top-K 集合一致：**{summary['ann_agreement']['all_topk_sets_equal']}**；"
              f"顺序一致：**{summary['ann_agreement']['all_topk_orders_equal']}**。见 ann_agreement.json，未为一致性调参。", "",
              "## Runtime latency", "",
              "GPU/库/处理器版本见 encoder_manifest.json。模型与 reference index 预加载后，每个 query 先 warmup 2 次，"
              "再按固定打乱顺序重复 20 次，共 140 个计时样本；不重新计算 reference embedding。"
              "图像文件为暖缓存，每次重新 decode；不包括模型加载、reference 预计算和 contact sheet 输出。", "",
              "|阶段（ms）|median|mean|p95|", "|---|---:|---:|---:|"]
    for stage in STAGES:
        v = lat["pooled"][stage]
        lines.append(f"|{stage}|{v['median']:.3f}|{v['mean']:.3f}|{v['p95']:.3f}|")
    lines += ["", "|Query（各 20 次，exact total ms）|median|mean|p95|",
              "|---|---:|---:|---:|"]
    for q in queries:
        v = lat["per_query"][q["query_round_id"]]["total_exact_ms"]
        lines.append(f"|{q['query_round_id']}|{v['median']:.3f}|{v['mean']:.3f}|{v['p95']:.3f}|")
    lines += ["", f"单次 exact query retrieval：median **{lat['pooled']['total_exact_ms']['median']:.3f} ms**，"
              f"p95 **{lat['pooled']['total_exact_ms']['p95']:.3f} ms**。"
              "forward 包含传输、CLS normalize、拷回 CPU 和 CUDA 同步。分项分位数不可直接相加；"
              "total_with_ann_validation 是额外同时运行 ANN 的总时间。每个 query 的 median/mean/p95 见 latency_summary.json。", "",
              "![latency](latency_breakdown.png)", "", "## 判断与证据边界", "",
              "判别规则在排名前固定并保存于 protocol.json：数据完整且 query≥5 时，Hit@5 和 Recall@5 均≥70% 为 STRONG；"
              "否则 Hit@20 和 Recall@20 均≥70% 为 MODERATE；否则为 WEAK。审计不完整或 query<5 为 UNRESOLVED。"
              "这些阈值是本次诊断的操作定义，不是统计显著性标准。", "",
              f"本次为 **{summary['judgment']}**；具体依据是上表固定 K 的命中/召回率。"
              "7 个 query 全来自同一连续 episode，彼此相关，且 teacher 重复引用了一些 step；"
              "因此本次标签只描述这组观测，不代表跨 episode 稳定性。", "",
              "Teacher citation 只表明 full-memory GPT 最终引用了它，不是所有有帮助经验的完整标注，"
              "也不证明引用的动作最优或确实因该经验而改变。Retriever 排在前面的未引用经验不能一概当作无用；"
              "反之找回引用也不证明可以正确 grounding Pa/Pb。没有因为结果改 encoder、crop、K、Graph prior 或启动下一阶段。", "",
              "## 文件", "",
              "- teacher_positive_manifest.json：query、原引用和 use（仅审计）。",
              "- reference_index_manifest.json：冻结图库、图像哈希、轨迹单位及 embedding/index 信息。",
              "- rankings.csv：7×139 完整 primary 排名；diversity_rankings.csv：secondary 排名。",
              "- per_query_metrics.csv、summary.json：全部指标与逐轮明细。",
              "- latency_samples.csv、latency_summary.json：140 个原始计时样本和统计。",
              "- contact_sheets/：每轮 query + Top-20 historical BEFORE，标注实际 teacher 引用。",
              "- protocol.json、encoder_manifest.json、input_hashes.json：固定配置、模型与输入审计。", ""]
    (out/"summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "run"))
    parser.add_argument("--episode", type=Path, default=EPISODE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return audit(args) if args.stage == "audit" else run(args)


if __name__ == "__main__":
    main()
