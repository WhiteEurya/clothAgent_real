"""Offline review of a shadow-gate episode; never changes execution or gate records."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path

from .experiments import one_shot_goal_planner_probe_v1 as io

CHOICES = ("TRUST_FAST", "NEED_DELIBERATION", "UNCERTAIN")
FINAL_CHOICES = ("FOLD_READY", "NOT_FOLD_READY", "UNCERTAIN")


def optional(path, default=None):
    return io.read(path) if path.exists() else default


def summarize(root, summary):
    """Build machine-readable statistics without treating Host rejects as bad actions."""
    fast, gates, host, rows = Counter(), Counter(), Counter(), []
    fast_times, gate_times, gate_valid_times, fast_action_times = [], [], [], []
    gate_statuses = Counter()
    for item in summary["rounds"]:
        folder = root/item["round"]
        plan = optional(folder/"fast_plan.json", {})
        fast_status = optional(folder/"fast_status.json", {})
        gate = optional(folder/"gate_result.json", {})
        status = optional(folder/"gate_status.json", {}).get("status", "NOT_RUN")
        validation = optional(folder/"host_validation.json", {})
        timing = item["timing"]
        host[validation.get("status", "NOT_RUN")] += 1
        if fast_status.get("valid"):
            fast[plan["decision"]] += 1
            fast_times.append(timing["fast_planner_wall_s"])
            if plan["decision"] == "ACTION":
                fast_action_times.append(timing["fast_planner_wall_s"])
        if plan.get("decision") != "ACTION" or not fast_status.get("valid"):
            continue
        gate_statuses[status] += 1
        if status in ("VALID", "ERROR", "INTERRUPTED"):
            gate_times.append(timing["gate_wall_s"])
        if status == "VALID":
            gates[gate["decision"]] += 1
            gate_valid_times.append(timing["gate_wall_s"])
        binding_files = ("fast_plan.json", "gate_result.json", "gate_status.json",
                         "before/rgb.png", "gate_input_overlay.png", "host_validation.json",
                         "execution.json", "after/rgb.png", "after/unavailable.json")
        bindings = {name: io.sha(folder/name) for name in binding_files if (folder/name).is_file()}
        rows.append(dict(round_id=folder.name, gate_decision=gate.get("decision"),
            gate_status=status, host_status=validation.get("status", "NOT_RUN"),
            execution_status=item["execution"]["status"], human_label=None,
            binding_sha256=bindings))
    fast_stats = io.stats(fast_times)
    result = dict(mode="SHADOW_ONLY", fast_decision_counts={k:fast[k] for k in ("ACTION","DONE","ABSTAIN")},
        gate_decision_counts={k:gates[k] for k in ("TRUST","DELIBERATE","UNCERTAIN")},
        gate_count_scope="Valid model responses only; synthetic error fallback excluded",
        gate_status_counts=dict(gate_statuses), host_reject_count=host["REJECT"],
        host_status_counts=dict(host), physical_execution_count=summary["physical_actions"],
        fast_planner_latency_s=fast_stats, fast_action_latency_s=io.stats(fast_action_times),
        gate_latency_s=io.stats(gate_times), gate_valid_latency_s=io.stats(gate_valid_times),
        total_fast_planner_wall_s=sum(r["timing"]["fast_planner_wall_s"] for r in summary["rounds"]),
        total_gate_wall_s=sum(r["timing"]["gate_wall_s"] for r in summary["rounds"]),
        gate_human_correspondence=rows, human_fold_ready=summary.get("human_fold_ready"),
        success_questions=dict(
            fast_20_25s=dict(status="OBSERVED_MEDIAN_ONLY" if fast_times else "NOT_MEASURED",
                median_s=fast_stats["median"],
                within_20_25_s=20 <= fast_stats["median"] <= 25 if fast_times else None,
                note="Gate excluded. Small, unpaired episodes cannot establish unchanged latency."),
            bad_action_detection="PENDING_HUMAN_REVIEW",
            normal_action_false_deliberate="PENDING_HUMAN_REVIEW",
            false_trust="PENDING_HUMAN_REVIEW",
            cheap_distillation="UNRESOLVED; labels and independent validation required; no training launched"))
    result["codex_recovery_enabled"] = summary.get("codex_recovery_enabled", False)
    result["codex_recovery_calls"] = summary.get("codex_recovery_calls", 0)
    result["codex_recovery_latency_s"] = io.stats([
        r["timing"]["deliberative_planner_wall_s"] for r in summary["rounds"]
        if r["timing"].get("deliberative_planner_wall_s", 0) > 0])
    summary["planner_layers"] = result
    return result


def render_review(root, summary):
    result = summary["planner_layers"]
    manifest = dict(schema_version=1, episode_id=str(root.resolve()),
        protocol_sha256=io.sha(root/"protocol.json"), goal_sha256=io.sha(root/"goal/target.png"),
        rounds=result["gate_human_correspondence"])
    io.save(root/"review_manifest.json", manifest)
    cards = []
    for row in manifest["rounds"]:
        rid = row["round_id"]
        images = []
        for label, relative in (
            ("Current RGB", f"{rid}/before/rgb.png"), ("Goal RGB", "goal/target.png"),
            ("FastPlanner proposal", f"{rid}/fast_action_overlay.png"), ("After（仅人工可见）", f"{rid}/after/rgb.png")):
            images.append(f'<figure><figcaption>{label}</figcaption>'+
                (f'<img src="{relative}" alt="{label}">' if (root/relative).exists() else
                 '<p>没有 after；未执行或采集已停止。不要把缺图当作无进展。</p>')+'</figure>')
        buttons = "".join(f'<label><input type="radio" name="{rid}" data-field="choice" '
                          f'value="{c}">{c}</label>' for c in CHOICES)
        cards.append(f'<section data-round="{rid}"><h2>{rid}</h2><p>'
            f'Gate: {html.escape(str(row["gate_decision"]))} ({row["gate_status"]}) · '
            f'Host: {row["host_status"]} · Execution: {row["execution_status"]}</p>'
            f'<div class="images">{"".join(images)}</div><div>{buttons}</div>'
            '<p>Fast action reasonable? <select data-field="reasonable"><option value="">未评</option>'
            '<option>YES</option><option>NO</option><option>UNCERTAIN</option></select> '
            'Actual progress? <select data-field="progress"><option value="">未评</option>'
            '<option>YES</option><option>NO</option><option>WORSE</option><option>UNCERTAIN</option>'
            '<option>NOT_EXECUTED</option></select></p>'
            '<textarea data-field="note" placeholder="人工备注"></textarea></section>')
    data = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    template = """<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Shadow gate · 人工评审</title><style>
body{font-family:system-ui;margin:24px;background:#101827;color:#e5e7eb}section{border:1px solid #475569;padding:18px;margin:24px 0}
.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}
label{margin-right:22px;display:inline-block}textarea{width:95%;height:65px}button,select{padding:8px;margin:8px}
.notice{color:#fbbf24}a{color:#7dd3fc}</style>
<h1>FastPlanner / Shadow gate 人工评审</h1>
<p class="notice">Gate 标签来自执行前独立请求；after 只供人工复盘。Host reject 不等于策略错误。
本页仅在浏览器保存人工意见并导出 JSON，不修改 gate、动作或执行记录。</p>
<p>先按“是否值得直接尝试”评动作，再单独记录 actual progress；不要只按结果倒推动作质量。</p>
<p>Reviewer <input id="reviewer"> 最终人工标签 <select id="final"><option value="">未评</option>
<option>FOLD_READY</option><option>NOT_FOLD_READY</option><option>UNCERTAIN</option></select>
<button id="export">导出人工评审 JSON</button><span id="status"></span></p>
__CARDS__
<script>
const manifest=__DATA__;
const key="shadow-review:"+manifest.episode_id+":"+manifest.protocol_sha256;
let saved={};try{saved=JSON.parse(localStorage.getItem(key)||"{}")}catch(e){}
document.querySelector("#reviewer").value=saved.reviewer||"";
document.querySelector("#final").value=saved.final_fold_ready||"";
for(const section of document.querySelectorAll("section")){
 const old=(saved.rounds||[]).find(r=>r.round_id===section.dataset.round)||{};
 for(const el of section.querySelectorAll("[data-field]")){
  const v=old[el.dataset.field]||"";
  if(el.type==="radio")el.checked=el.value===v;else el.value=v;
 }
}
function collect(){
 return {schema_version:1,episode_id:manifest.episode_id,protocol_sha256:manifest.protocol_sha256,
 goal_sha256:manifest.goal_sha256,reviewer:document.querySelector("#reviewer").value,
 final_fold_ready:document.querySelector("#final").value||null,
 rounds:manifest.rounds.map(row=>{
  const s=document.querySelector('[data-round="'+row.round_id+'"]');
  return {round_id:row.round_id,binding_sha256:row.binding_sha256,
   choice:s.querySelector('input:checked')?.value||null,
   reasonable:s.querySelector('[data-field="reasonable"]').value||null,
   progress:s.querySelector('[data-field="progress"]').value||null,
   note:s.querySelector("textarea").value};
 })};
}
document.addEventListener("input",()=>{try{localStorage.setItem(key,JSON.stringify(collect()));
 document.querySelector("#status").textContent="已保存到本浏览器"}catch(e){
 document.querySelector("#status").textContent="本地保存不可用，请导出 JSON"}});
document.querySelector("#export").onclick=()=>{
 const url=URL.createObjectURL(new Blob([JSON.stringify(collect(),null,2)],{type:"application/json"}));
 const a=document.createElement("a");a.href=url;a.download="shadow_gate_human_review.json";a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></html>"""
    (root/"review.html").write_text(template.replace("__CARDS__", "".join(cards)).replace("__DATA__", data))


def append_report(root, summary):
    s = summary["planner_layers"]
    text = [
        "\n## Planner layers / shadow gate\n",
        "执行来源始终是 FastPlanner。PreActionGate 是独立预执行请求，只记标签；"
        "DELIBERATE/UNCERTAIN/请求错误均不替换动作。MemoryPlanner 禁用。"
        f"独立 Host-error Codex recovery：{s.get('codex_recovery_enabled', False)}；"
        f"调用 {s.get('codex_recovery_calls', 0)} 次；耗时 {s.get('codex_recovery_latency_s')}。"
        "Codex 轮次不计入 FastPlanner/gate 人工对照；原拒绝动作没有执行，其 after 不以恢复动作结果替代。",
        f"Fast 决策：{s['fast_decision_counts']}；有效 gate 决策：{s['gate_decision_counts']}。",
        f"Gate 状态：{s['gate_status_counts']}。错误 fallback UNCERTAIN 不混入有效模型标签。",
        f"Host reject：{s['host_reject_count']}；physical execution：{s['physical_execution_count']}。",
        f"FastPlanner latency（不含 gate）：{s['fast_planner_latency_s']}。",
        f"Fast ACTION-only latency：{s['fast_action_latency_s']}。",
        f"Gate latency（含错误尝试，单列）：{s['gate_latency_s']}；有效返回：{s['gate_valid_latency_s']}。",
        f"Fast 总 wall {s['total_fast_planner_wall_s']:.2f}s；gate 总 wall {s['total_gate_wall_s']:.2f}s。",
        f"1. Fast 是否仍在20–25s：{s['success_questions']['fast_20_25s']}。",
        "2. 明显 bad action 能否检出：待人工评审。",
        "3. 正常 action 是否大量误报 DELIBERATE：待人工评审。",
        "4. false-TRUST：未评，不能填0或拿 Host reject 代替人工标签。",
        "5. 是否值得蒸馏到<1s：UNRESOLVED；没有启动训练或第二阶段。",
        "最终人工 FOLD_READY：未判定。评审导出的 final_fold_ready 另存，不回写执行记录。",
        "[人工评审页面](review.html) · [逐轮 gate / human 对照](review_manifest.json)",
        "\n|Round|Gate|Gate status|Host|Human|\n|---|---|---|---|---|"]
    text += [f"|{r['round_id']}|{r['gate_decision']}|{r['gate_status']}|{r['host_status']}|PENDING|"
             for r in s["gate_human_correspondence"]]
    with (root/"report.md").open("a") as stream:
        stream.write("\n".join(text)+"\n")


def evaluate_review(root, ratings, output):
    """Bind labels to the exact pre-action evidence; emit a separate analysis directory."""
    manifest = io.read(root/"review_manifest.json")
    summary = io.read(root/"episode_summary.json")
    if summary.get("termination_reason") is None:
        raise ValueError("Review analysis requires a finished episode")
    labels = io.read(ratings)
    for key in ("episode_id", "protocol_sha256", "goal_sha256"):
        if labels.get(key) != manifest[key]:
            raise ValueError("Review belongs to different episode/evidence: "+key)
    if io.sha(root/"protocol.json") != manifest["protocol_sha256"] or io.sha(root/"goal/target.png") != manifest["goal_sha256"]:
        raise ValueError("Episode protocol/goal changed")
    if labels.get("final_fold_ready") not in (*FINAL_CHOICES, None):
        raise ValueError("Invalid final human label")
    originals = {r["round_id"]: r for r in manifest["rounds"]}
    for rid, original in originals.items():
        if any(io.sha(root/rid/name) != value for name,value in original["binding_sha256"].items()):
            raise ValueError("Source evidence changed")
        actual_gate = optional(root/rid/"gate_result.json", {}).get("decision")
        actual_status = optional(root/rid/"gate_status.json", {}).get("status", "NOT_RUN")
        if (original["gate_decision"], original["gate_status"]) != (actual_gate, actual_status):
            raise ValueError("Review manifest differs from saved gate")
    matrix = {g:{h:0 for h in CHOICES} for g in ("TRUST","DELIBERATE","UNCERTAIN")}
    seen, paired, reviewed, rows = set(), 0, 0, []
    for label in labels.get("rounds", []):
        rid = label["round_id"]
        if rid in seen or rid not in originals:
            raise ValueError("Unknown/duplicate review round")
        seen.add(rid); original=originals[rid]
        if label.get("binding_sha256") != original["binding_sha256"]:
            raise ValueError("Review round evidence mismatch")
        if any(io.sha(root/rid/name) != value for name,value in original["binding_sha256"].items()):
            raise ValueError("Source evidence changed")
        choice=label.get("choice")
        if choice not in (*CHOICES, None):
            raise ValueError("Invalid human choice")
        if label.get("reasonable") not in (None,"YES","NO","UNCERTAIN"):
            raise ValueError("Invalid reasonable label")
        if label.get("progress") not in (None,"YES","NO","WORSE","UNCERTAIN","NOT_EXECUTED"):
            raise ValueError("Invalid progress label")
        reviewed += choice is not None
        if choice and original["gate_status"]=="VALID":
            matrix[original["gate_decision"]][choice] += 1
            paired += 1
        rows.append(dict(**original, review=label))
    bad=sum(matrix[g]["NEED_DELIBERATION"] for g in matrix)
    normal=sum(matrix[g]["TRUST_FAST"] for g in matrix)
    false_trust=matrix["TRUST"]["NEED_DELIBERATION"]
    false_deliberate=matrix["DELIBERATE"]["TRUST_FAST"]
    trust_known=matrix["TRUST"]["TRUST_FAST"]+false_trust
    def rate(a,b): return a/b if b else None
    report=dict(episode_id=manifest["episode_id"],reviewer=labels.get("reviewer"),
        final_fold_ready=labels.get("final_fold_ready"),action_rounds=len(originals),reviewed_rounds=reviewed,
        pending_rounds=len(originals)-reviewed,valid_gate_human_pairs=paired,
        decisive_human_gate_pairs=bad+normal,
        confusion_matrix=matrix,normal_action_count=normal,bad_action_count=bad,
        false_trust_count=false_trust if bad+normal else None,
        false_trust_over_human_bad=rate(false_trust,bad),
        false_trust_over_decisively_reviewed_gate_trust=rate(false_trust,trust_known),
        false_deliberate_count=false_deliberate if bad+normal else None,
        false_deliberate_over_human_normal=rate(false_deliberate,normal),
        bad_action_deliberate_recall=rate(matrix["DELIBERATE"]["NEED_DELIBERATION"],bad),
        fast_planner_latency_s=summary["planner_layers"]["fast_planner_latency_s"],
        gate_latency_s=summary["planner_layers"]["gate_latency_s"],
        distillation_decision="UNRESOLVED; descriptive evidence only, no automatic next stage",
        policy_labels_from_host_reject=False,rows=rows)
    output.mkdir(parents=True, exist_ok=False)
    io.save(output/"human_review.json",labels)
    io.save(output/"review_summary.json",report)
    text=["# Shadow gate 人工评审汇总","",
        f"人工已评 {reviewed}/{len(originals)}；有效 gate 配对 {paired}。最终人工标签：{report['final_fold_ready']}。",
        f"FastPlanner latency：{report['fast_planner_latency_s']}；gate：{report['gate_latency_s']}。",
        f"Bad action 标为 DELIBERATE 的比例：{report['bad_action_deliberate_recall']}（分母 {bad}）。",
        f"正常 action 误报 DELIBERATE：{report['false_deliberate_count']}，比例 {report['false_deliberate_over_human_normal']}（分母 {normal}）。",
        f"false-TRUST：{report['false_trust_count']}；占人工 bad：{report['false_trust_over_human_bad']}；"
        f"占已明确评审 gate TRUST：{report['false_trust_over_decisively_reviewed_gate_trust']}。",
        "UNCERTAIN 单列；未评不当作正确，gate 请求错误不当作模型判断，Host reject 不当作 policy failure。",
        "是否值得蒸馏：UNRESOLVED。以上为单次评审描述，没有自动开启训练、routing 或下一阶段。",
        "原 episode 的 plan、gate、execution、summary 均未修改。",
        "\n|Gate \\ Human|TRUST_FAST|NEED_DELIBERATION|UNCERTAIN|\n|---|---:|---:|---:|"]
    text += [f"|{g}|{matrix[g]['TRUST_FAST']}|{matrix[g]['NEED_DELIBERATION']}|{matrix[g]['UNCERTAIN']}|" for g in matrix]
    (output/"report.md").write_text("\n".join(text)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",type=Path,required=True)
    parser.add_argument("--ratings",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    evaluate_review(args.episode.resolve(),args.ratings,args.output)


if __name__=="__main__":
    main()
