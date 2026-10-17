"""Paired offline full/minimal e001 GPT grounding. Never constructs a runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import math
from pathlib import Path
import statistics
import time

from PIL import Image,ImageDraw
from . import graph_policy_replay as pr
from .graph_policy import load_frozen,instantiate
from .graph_minimal_grounding import read,write,sha,compile_e001,MinimalGrounder,validate,SPEC_E001
from .discovery_model import CodexMultimodal
from cloth_agent.dual_arm.observation import Observation

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/minimal_edge_grounding'
GRAPH=ROOT/'runs/graph_policy_full02/policy_graph_v2.json'
SOURCE=ROOT/'reports/action_anchor_transfer/manifest.json'


def prepare():
    source=read(SOURCE);OUT.mkdir(exist_ok=True)
    compile_e001(GRAPH,Path(source['reference']['dispatch_file']).parent,OUT/'host_templates.json')
    manifest=dict(model='gpt-6-astra',graph=str(GRAPH),graph_sha256=sha(GRAPH),queries=source['queries'],
                  source_manifest_sha256=sha(SOURCE),host_templates_sha256=sha(OUT/'host_templates.json'),
                  protocol='Fresh paired full/minimal, one request per condition plus existing bounded format repair; alternating condition order, two sample workers. No runtime or robot.',
                  old_coordinates='Noisy historical GPT reference, never input to the two model conditions',spec=SPEC_E001)
    if (OUT/'manifest.json').exists():assert read(OUT/'manifest.json')==manifest
    else:write(OUT/'manifest.json',manifest)
    return manifest


def query(row,index,manifest):
    graph=load_frozen(GRAPH);edge=next(e for e in graph['edges'] if e['edge_id']=='e001')
    obs=Observation.load(Path(row['image']).parent)
    assert sha(obs.image)==row['sha256']
    for mode in (['full','minimal'] if index%2==0 else ['minimal','full']):
        folder=OUT/'samples'/row['sample_id']/mode;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'result.json').exists():continue
        requests=pr.policy_requests(folder/'model',CodexMultimodal(manifest['model'],900),no_cache=True,max_calls=2,
                                    grounding_mode=mode,grounding_templates=OUT/'host_templates.json' if mode=='minimal' else None)
        r=dict(sample_id=row['sample_id'],mode=mode,status='ERROR',Pa_pred=None,Pb_pred=None,Pa_gt=row['Pa'],Pb_gt=row['Pb'],
               host_status='NOT_ATTEMPTED',confidence=None)
        start=time.perf_counter()
        try:
            if mode=='full':
                candidate=requests.ask('ground_edge',pr.grounding_payload(graph,edge,obs),[('current',obs.image)],validator=pr.validate_grounding)
                write(folder/'grounding.json',candidate)
                if candidate['status']=='READY':r.update(status='OK',Pa_pred=candidate['slots'][0]['pixel_xy'],Pb_pred=candidate['slots'][-1]['pixel_xy'])
                else:r.update(status=candidate['status'])
            else:
                value=requests.ask('ground_edge_minimal',SPEC_E001,[('current',obs.image)],validator=lambda v:validate(v,(1280,720)))
                write(folder/'minimal_anchors.json',value)
                r.update(status='OK' if value['grasp_px'] is not None and value['target_px'] is not None else 'UNRESOLVED',Pa_pred=value['grasp_px'],Pb_pred=value['target_px'],confidence=value['confidence'])
            r['gpt_wall_s']=time.perf_counter()-start
            # Separate deterministic Host timing and failure from model latency/format success.
            host=time.perf_counter()
            try:
                if mode=='minimal':candidate=MinimalGrounder(OUT/'host_templates.json').expand(value,edge,obs,folder);write(folder/'grounding.json',candidate)
                pr.validate_grounding(candidate)
                side=pr.validate_sleeve_side(candidate,edge,obs);write(folder/'garment_side_check.json',side)
                _,source,measurements=instantiate(pr.primitive_candidate(candidate),[edge],obs)
                (folder/'motion_offline_only.py').write_text(source);write(folder/'surface_measurements.json',measurements)
                r['host_status']='PIXEL_DEPTH_TEMPLATE_OK_NO_IK_OR_EXECUTION'
            except Exception as exc:r.update(host_status='REJECT',host_error=f'{type(exc).__name__}: {exc}')
            r['host_elapsed_s']=time.perf_counter()-host
        except Exception as exc:r.update(error=f'{type(exc).__name__}: {exc}',gpt_wall_s=time.perf_counter()-start)
        logs=[json.loads(s) for s in (folder/'model/requests.jsonl').read_text().splitlines()] if (folder/'model/requests.jsonl').exists() else []
        r['model_calls']=requests.model_calls;r['request_latency_s']=sum(l.get('elapsed_s',0) for l in logs)
        r['first_attempt_failed']=bool(logs and 'error' in logs[0]);r['final_structured_failure']=not any('structured_response' in l for l in logs)
        r['prompt_chars']=len(logs[0]['prompt']) if logs else None
        r['response_chars']=len(logs[-1].get('raw_response','')) if logs else None
        for k,p,g in [('grasp','Pa_pred','Pa_gt'),('target','Pb_pred','Pb_gt')]:r[k+'_deviation_px']=math.dist(r[p],r[g]) if r[p] is not None else None
        write(folder/'result.json',r);print(mode,row['sample_id'],r['status'],round(r['gpt_wall_s'],1),'s',r['host_status'],flush=True)


def run():
    m=prepare()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(query,r,i,m) for i,r in enumerate(m['queries'])]
        for f in as_completed(futures):f.result()
    report()


def report():
    m=read(OUT/'manifest.json');rows=[];summary={}
    for q in m['queries']:
        for mode in ['full','minimal']:
            f=OUT/'samples'/q['sample_id']/mode/'result.json'
            if f.exists():rows.append(read(f))
    for mode in ['full','minimal']:
        rs=[r for r in rows if r['mode']==mode]
        if not rs:continue
        s=dict(n=len(rs),structured_failures=sum(r['final_structured_failure'] for r in rs),first_attempt_failures=sum(r['first_attempt_failed'] for r in rs),
               mean_latency_s=statistics.mean(r['gpt_wall_s'] for r in rs),median_latency_s=statistics.median(r['gpt_wall_s'] for r in rs),
               median_prompt_chars=statistics.median(r['prompt_chars'] for r in rs if r['prompt_chars'] is not None),
               host_success=sum(r['host_status']=='PIXEL_DEPTH_TEMPLATE_OK_NO_IK_OR_EXECUTION' for r in rs))
        for k in ['grasp','target']:
            vals=[r[k+'_deviation_px'] for r in rs if r[k+'_deviation_px'] is not None]
            s[k+'_median_deviation_px']=statistics.median(vals) if vals else None
        summary[mode]=s
    write(OUT/'results.json',rows);write(OUT/'summary.json',summary)
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (OUT/'predictions.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    # Paired overlays, all original-image coordinates (display at half size).
    panels=[]
    for q in m['queries']:
        canvas=Image.new('RGB',(2560,820),'white');d=ImageDraw.Draw(canvas)
        for i,mode in enumerate(['full','minimal']):
            r=next((r for r in rows if r['mode']==mode and r['sample_id']==q['sample_id']),None)
            canvas.paste(Image.open(q['image']).convert('RGB'),(i*1280,100))
            d.text((i*1280+10,10),mode+' '+q['sample_id']+' '+str(r['status'] if r else 'PENDING'),fill='black')
            if not r:continue
            d.text((i*1280+10,35),f"{r['gpt_wall_s']:.1f}s | red=grasp cyan=target; yellow X=historical GPT, not truth",fill='black')
            for key,gt,color in [('Pa_pred','Pa_gt','red'),('Pb_pred','Pb_gt','cyan')]:
                if r[key]:
                    x,y=r[key];x+=i*1280;y+=100;d.ellipse([x-9,y-9,x+9,y+9],outline=color,width=4)
                x,y=r[gt];x+=i*1280;y+=100;d.line([x-9,y-9,x+9,y+9],fill='yellow',width=4);d.line([x-9,y+9,x+9,y-9],fill='yellow',width=4)
        folder=OUT/'overlays';folder.mkdir(exist_ok=True);canvas.save(folder/(q['sample_id']+'.png'));panels.append(canvas.resize((1280,410)))
    montage=Image.new('RGB',(1280,410*len(panels)),'white')
    for i,p in enumerate(panels):montage.paste(p,(0,i*410))
    montage.save(OUT/'montage.jpg');print(json.dumps(summary,indent=2),flush=True)
    write_report(rows,summary)


def write_report(rows,summary):
    if set(summary)!={'full','minimal'}:return
    lines=['# Minimal edge grounding offline ablation','',
           '## Scope and protocol',
           'Seven historical e001 / S1 query RGBs, separate from the successful template reference. Fresh GPT-6-astra calls for BOTH modes on identical images; alternating A/B order, two concurrent sample workers, cache disabled. Same transport/model and existing maximum one format repair. No robot runtime constructed, no execution, no training. One trial per image/mode; service jitter is not controlled and these are not repeated latency measurements.',
           'Archived GPT grasp/target are noisy consistency references only. Neither mode receives them. Fresh full-mode outputs are also not human truth. All seven are the same T-shirt, camera and narrow orientation range.','',
           '## What full mode actually sends',
           'Current RGB + task subgoal/side convention + ownership instructions + selected edge ID/family/full action template + primitive method/index sequence + grasp-offset metadata + image size + observation geometry (calibration, transforms, joints, etc.). The long prompt asks for per-primitive pixel, surface-relative height, world RPY, garment collar/hem frame, evidence, explanation, risks, violations and missing parameters.',
           'It does NOT send the whole graph, other nodes, outcome/failure history or complete seed XYZ trajectory. It already forbids free planning; Host already emits motion Python. The optimization removes redundant parameter completion rather than fixing an existing free-planner call.','',
           '## Minimal mode',
           'One current RGB and the short e001 grasp/target spec; only grasp_px, target_px, confidence returned. Null coordinates allow explicit uncertainty. No family selection, trajectory, XYZ, heights, orientation, frame landmarks, graph context or outcome reasoning in the model request. A short garment-side convention remains essential: mentally rotate collar up without mirroring.',
           'Prompt characters include the existing schema/JSON wrapper; they are not token counts.','',
           '| Mode | n | Mean / median GPT wall seconds | Median grasp deviation px | Median target deviation px | Final structured failures | Prompt chars (median) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for mode,s in summary.items():
        lines.append(f"| {mode} | {s['n']} | {s['mean_latency_s']:.2f} / {s['median_latency_s']:.2f} | {s['grasp_median_deviation_px']:.2f} | {s['target_median_deviation_px']:.2f} | {s['structured_failures']}/{s['n']} | {s['median_prompt_chars']:.0f} |")
    a,b=summary['full'],summary['minimal']
    paired=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        pair={r['mode']:r for r in rows if r['sample_id']==sid}
        if set(pair)=={'full','minimal'}:
            paired.append(dict(sample_id=sid,**{k+'_deviation_px':math.dist(pair['full'][p],pair['minimal'][p]) if pair['full'][p] and pair['minimal'][p] else None for k,p in [('grasp','Pa_pred'),('target','Pb_pred')]}))
    write(OUT/'paired_full_minimal_deviation.json',paired)
    lines += ['',f"Median latency reduction: {(1-b['median_latency_s']/a['median_latency_s'])*100:.1f}%; mean latency reduction: {(1-b['mean_latency_s']/a['mean_latency_s'])*100:.1f}%. First-attempt failures: full {a['first_attempt_failures']}, minimal {b['first_attempt_failures']}. All parse errors remain explicit, not converted to unknown coordinates.",
              '', '## Host-owned parameters and remaining validators',
              'The opt-in Host bundle compiles the existing grasp + four move_tcp skeleton once from a successful historical e001 grounding: normalized along/across waypoint geometry and surface-relative height profile [0,60,120,60,0] mm. Reference RPY is checked against the constant seed orientation. Current grasp/target instantiate that profile deterministically.',
              'Host obtains current collar/hem estimates using SIFT torso-print correspondences and partial-affine RANSAC from the reference. It requires unique inliers with spatial support and bounded scale/rotation; it does not copy stale frame coordinates on failure. Current RGB-D collar/hem axis determines a common world-Z yaw rotation of the frozen RPY profile. This is a new deterministic perception adapter, not an independent semantic guarantee.',
              'Existing Observation.sample and instantiate handle pixel-to-XYZ and surface Z. Existing sleeve-side check is called unchanged. Existing grasp lowering, parallel-jaw orientation handling, release/Home behavior, workspace, IK, collision and pre-dispatch safety remain in the original runtime. We only exercised offline frame/depth/slot generation, not IK, collision trajectories or physical success.',
              f"Offline Host expansion + depth checks passed: full {a['host_success']}/{a['n']}, minimal {b['host_success']}/{b['n']}. Host timing and registration evidence are saved separately from model latency.",
              'Missing reference registration, low-confidence/null anchors, changed templates or unsupported edges cause explicit instantiation failure; no second GPT frame request and no silent full/free-planner fallback. The reduced output omits the full-mode general visual violation/risk explanation; it is not semantically identical to that audit.',
              '', '## Scope of switching',
              'Old implementation remains the default (--grounding-mode full). Both ordinary Graph Replay and online-growth graph actions share the opt-in minimal route. Only e001 is compiled and validated in this first bundle; selecting e000/e002/etc. in minimal mode will stop explicitly. This bundle is NOT an end-to-end whole-graph replacement. The current SIFT adapter is limited to same-camera, same-garment, modest pose changes; hem/side-fold templates are not silently extrapolated.',
              'Add these options to a replay command to select the mode (no execution performed here):','```bash',
              '--grounding-mode minimal \\',
              '  --grounding-templates reports/minimal_edge_grounding/host_templates.json','```',
              'Offline ablation (existing per-sample results are reused; no hardware):','```bash',
              '/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.minimal_grounding_ablation run','```',
              '', '## Artifacts and implementation',
              '- graph_policy_replay.py: opt-in routing and CLI; legacy validators and execution route retained.',
              '- graph_minimal_grounding.py: tiny model contract, frozen Host bundle, deterministic frame/waypoint adapter.',
              '- minimal_grounding_ablation.py: paired offline runs, latency/deviation/error records and overlays.',
              '- tests/test_graph_minimal_grounding.py: contract, no full-context leakage, profile/orientation, rejection and replay integration tests.',
              '- manifest.json, host_templates.json and checksum: data/model/template provenance; original graph untouched.',
              '- samples/*/{full,minimal}/model/requests.jsonl: exact actual prompts, responses, elapsed time and call metadata.',
              '- predictions.csv, results.json, summary.json, montage.jpg and overlays/: per-query comparisons.',
              '', '## Visual region review',
              'See semantic_review.json for the completed assistant visual review of the paired overlays. This is qualitative region inspection, NOT independent human ground truth or execution validation.']
    if (OUT/'tests.log').exists():lines += ['', '## Tests','```',(OUT/'tests.log').read_text().strip(),'```']
    if (OUT/'semantic_review.json').exists():
        review=read(OUT/'semantic_review.json')
        lines += ['', 'Completed visual review: '+str(len(review['samples']))+' image/mode pairs; no obvious wrong-side or background anchors observed. Minimal target locations vary within the receiving region; this is qualitative plausibility, not a numerical semantic accuracy score.']
    lines += ['', 'Direct fresh full→minimal consistency (separate from archived-reference deviations):',
              '```json',json.dumps({k:statistics.median(r[k] for r in paired if r[k] is not None) for k in ['grasp_deviation_px','target_deviation_px']},indent=2),'```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','report']);a=ap.parse_args();{'prepare':prepare,'run':run,'report':report}[a.stage]()
