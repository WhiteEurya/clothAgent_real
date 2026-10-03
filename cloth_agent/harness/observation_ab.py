"""Controlled observation-method A/B: different image preparation, identical reasoning."""
from __future__ import annotations

import argparse
import html
import time
from pathlib import Path

from .common import canonical, digest, read_json, write_json
from .information_flow import EXECUTORS, REQUESTS, ObservationHost
from .information_probe import ROOT, ProbeBackend, compact_skill, tool_metrics
from .model import RuntimeClaude
from .planning_probe import load_case, validate_plan
from .policy import PolicyError, obj, validate_schema
from ..planner_backend import parse_claude_json

FREE_SCHEMA = obj({'selected_image_ids': {'type':'array','maxItems':6,'uniqueItems':True,
                                          'items':{'type':'string'}}})
SKILL_SCHEMA = obj({'observation_requests': REQUESTS})
OBSERVE_PROMPT = (
    'Prepare useful views for a later independent visual planner solving the supplied current task. '
    'Inspect the current garment and semantic reference images. Resolve which views can reveal relevant '
    'orientation, target-side boundaries and marker correspondence. Do not select a grasp, write a motion '
    'plan or return scene conclusions. Original roots will always be available to the later planner. '
    'Avoid unnecessary edits; crops/enlargement do not recover occluded or missing sensor detail. '
)
REASON_PROMPT = (
    'Perform the supplied original visual-planning task using the actual attached images. '
    'Original roots and additional observation views are attached together; no file or image-tool calls '
    'are needed or available. Additional views carry only host-verified image provenance, not scene conclusions. '
    'image_id indexes the attachments. original_image_id and to_original map a derived view to its root. '
    'Rxxx markers retain their original identities. Reference images provide semantic goals only, never '
    'executable coordinates. Use the same original output schema and current registry. '
    'Explain uncertainty within that contract; do not claim RGB proves single-ply isolation or physical success.\n'
)


def shared_context(case):
    return {'current_goal':case['fold_goal'], 'original_prompt':case['prompt'],
            'original_system_prompt':case['system_prompt'], 'context':case['context'],
            'candidate_registry':case['registry']}


def roots_catalog(case):
    return [{**item,'original_image_id':item['image_id'],'to_original':[1,0,0,0,1,0]}
            for item in case['observation']['images']]


def normalized_catalog(case, views):
    """Only geometric provenance crosses the observation/reasoning boundary."""
    roots = roots_catalog(case)
    by_id = {item['image_id']:item for item in roots}
    catalog = list(roots)
    for view in views:
        root = by_id[view['original_image_id']]
        catalog.append({'image_id':f'image_{len(catalog)}','original_image_id':root['image_id'],
                        'role':root['role'],'reference_kind':root.get('reference_kind'),
                        'size':view['size'],'to_original':view['to_original']})
    return catalog


def prepare_free(case, images, backend, output, *, model_name, timeout, max_edits):
    output.mkdir(parents=True,exist_ok=False)
    payload = {**shared_context(case),'images':roots_catalog(case)}
    prompt = OBSERVE_PROMPT + (
        'Choose image operations yourself. Available tools are mcp__cloth_image__list_images, '
        'view_image, crop_image, rotate_image, resize_image, image_info and map_point. '
        'Call them directly by their full mcp__cloth_image__ names; ToolSearch and Bash are not enabled. '
        'Return selected_image_ids listing only the useful final derived views, at most six. '
        'Omit intermediate views and original roots; return [] if originals suffice. Use StructuredOutput '
        'with the supplied schema. Do not invent image IDs.\n') + canonical(payload)
    response = backend.invoke(prompt=prompt,image_paths=images,schema=FREE_SCHEMA,
        system_prompt='Observe supplied images only; prepare views, no grasp selection or robot execution.',
        model=model_name,max_turns=16,image_edit_limit=max_edits,overall_timeout_s=timeout,
        debug_dir=output/'trace',usage_run_dir=output,usage_stage='free_observation')
    (output/'stdout.jsonl').write_text(response.stdout)
    (output/'stderr.txt').write_text(response.stderr)
    result = parse_claude_json(response.stdout)
    write_json(output/'result.json',result)
    validate_schema(result,FREE_SCHEMA)
    debug = read_json(output/'trace/image_debug.json')
    available = {v['image_id']:v for v in debug['views']}
    paths, views = list(images), []
    for image_id in result['selected_image_ids']:
        if image_id not in available:
            raise PolicyError('Observer selected an unavailable image')
        view = available[image_id]
        if not view.get('operation'):  # shared roots are always included exactly once
            continue
        paths.append(Path(view['path']))
        views.append({'original_image_id':f'image_{view["original_image_index"]}',
                      'size':view['size'],'to_original':view['to_original']})
    return paths,normalized_catalog(case,views),tool_metrics(response.stdout,output/'trace')


def prepare_with_skill(case, images, artifact, model, output, *, timeout, max_edits):
    output.mkdir(parents=True,exist_ok=False)
    payload = {**shared_context(case),'images':roots_catalog(case),
               'skill':compact_skill(artifact),'host_executors':EXECUTORS,'max_host_edits':max_edits}
    prompt = OBSERVE_PROMPT + (
        'Use the existing observation methods to choose and bind at most three supported requests now. '
        'Host will execute them without model calls between operations. Return ONLY observation_requests. '
        'Each request names a visual gap and expected information gain, not a claimed result. '
        'For orientation use current clean root, roi=null, nonzero quarter-turn and enlarge=false; '
        'Host rotates clean and overlay together (two edits). For local_boundary use normalized clean-root '
        'ROI and degrees_clockwise=0, optionally enlarge (one or two edits). For overlay_occlusion '
        'Host crops both aligned roots and optionally enlarges both (two or four edits). '
        'No duplicate view requests. If originals suffice or editing cannot resolve the gap, return []. '
        'Do not output task-state facts, candidate IDs or scene conclusions. Be concise.\n') + canonical(payload)
    t=time.monotonic()
    result=model.invoke(prompt=prompt,schema=SKILL_SCHEMA,images=images,output=output/'bind',
                        stage='skill_observation',timeout_s=timeout)
    write_json(output/'result.json',result)
    validate_schema(result,SKILL_SCHEMA)
    if time.monotonic()-t >= timeout:
        raise TimeoutError('Observation budget exhausted')
    host=ObservationHost(case['observation'],images,artifact,output/'host',max_ops=max_edits)
    # These are requests for information, not assertions that it was acquired.
    gaps=[{'id':r['gap_id'],'status':'UNKNOWN'} for r in result['observation_requests']]
    stop=host.execute(result['observation_requests'],gaps)
    if stop: raise PolicyError(stop)
    return host.paths,normalized_catalog(case,host.catalog[len(images):]),{
        'image_edit_calls':0,'host_image_ops':host.ops,'host_execution_s':host.elapsed_s}


def reason_common(case, paths, catalog, model, output, *, timeout):
    """Exactly the same function, text, context, schema and tools for both arms."""
    contract={'instruction':REASON_PROMPT,'shared_context':shared_context(case),'schema':case['schema'],
              'model_configuration':model.configuration,'reasoning_budget_s':timeout}
    bundle={**shared_context(case),'images':catalog}
    write_json(output.parent/'reasoning_contract.json',contract)
    write_json(output.parent/'reasoning_bundle.json',bundle)
    result=model.invoke(prompt=REASON_PROMPT+canonical(bundle),schema=case['schema'],images=paths,
                        output=output,stage='common_visual_reasoning',timeout_s=timeout)
    write_json(output.parent/'raw_reasoning_result.json',result)
    validation=validate_plan(result,case)
    write_json(output.parent/'result.json',result)
    return result,validation,digest(contract)


def run_arm(case, images, artifact, output, *, mode, observer_backend, observer_model,
            reasoning_model, model_name, timeout=720, observation_timeout=360, max_edits=6):
    if not 0 < observation_timeout < timeout:
        raise ValueError('Reserve equal positive observation and reasoning stage budgets for both arms')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={'status':'RUNNING','mode':mode,'observation_s':0.0,'reasoning_s':0.0,
            'common_context_hash':case['context_hash'],'observation_hash':digest(case['observation']),
            'model_requested':model_name,'quality':'NOT_INDEPENDENTLY_VERIFIED'}
    write_json(output/'report.json',report)
    try:
        print('[observation] '+mode,flush=True)
        t=time.monotonic()
        try:
            if mode=='free':
                paths,catalog,metrics=prepare_free(case,images,observer_backend,output/'observe',
                    model_name=model_name,timeout=min(timeout,observation_timeout),max_edits=max_edits)
            elif mode=='skill':
                paths,catalog,metrics=prepare_with_skill(case,images,artifact,observer_model,output/'observe',
                    timeout=min(timeout,observation_timeout),max_edits=max_edits)
            else: raise ValueError('Unknown observation mode')
        finally:
            report['observation_s']=time.monotonic()-t
        report.update(metrics)
        write_json(output/'prepared_images.json',{'paths':[str(p) for p in paths],'catalog':catalog})
        # Both reasoning calls get exactly the same budget; a faster observer must
        # not buy extra reasoning time. Stage budgets exclude local bookkeeping.
        remaining=timeout-observation_timeout
        print('[same reasoning] '+mode,flush=True)
        t=time.monotonic()
        try:
            result,validation,contract_hash=reason_common(case,paths,catalog,reasoning_model,
                                                         output/'reason',timeout=remaining)
        finally:
            report['reasoning_s']=time.monotonic()-t
        report.update(status='COMPLETED',selected_reference=result['selected_reference'],
                      confidence=result['confidence'],validation=validation,
                      reasoning_contract_hash=contract_hash)
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        report.update(elapsed_s=time.monotonic()-start,observer_calls=list(observer_model.calls),
                      reasoning_calls=list(reasoning_model.calls))
        write_json(output/'report.json',report)
    return report


def render(output, report):
    esc=lambda v:html.escape(str(v))
    body='<h1>只改变观察方法，使用相同后续推理</h1><p>观察 A：Claude 自选工具；观察 B：skill 绑定与 Host 执行。后续均调用 reason_common，实际图片集中输入，均无图像编辑工具。观察阶段的判断文本不传入后续。</p><table><tr><th>组别</th><th>状态</th><th>观察 s</th><th>相同推理 s</th><th>总耗时 s</th><th>选点</th></tr>'
    for mode,a in report['arms'].items():
        body+=f'<tr><td>{mode}</td><td>{a["status"]}</td><td>{a["observation_s"]:.1f}</td><td>{a["reasoning_s"]:.1f}</td><td>{a["elapsed_s"]:.1f}</td><td>{esc(a.get("selected_reference",{}).get("reference_id","—"))}</td></tr>'
    body+='</table>'
    for mode,a in report['arms'].items():
        body+=f'<h2>{mode}</h2>'
        if a.get('error'):body+='<pre>'+esc(a['error'])+'</pre>'
        p=output/mode/'prepared_images.json'
        if p.exists():
            package=read_json(p);body+='<div class="images">'
            for path,item in zip(package['paths'],package['catalog']):
                rel=Path(path).resolve().relative_to(output.resolve()).as_posix()
                body+=f'<figure><a href="{esc(rel)}"><img src="{esc(rel)}"></a><figcaption>{esc(item["image_id"])} · {esc(item["role"])}</figcaption></figure>'
            body+='</div>'
        p=output/mode/'result.json'
        if p.exists():body+='<pre>'+esc(canonical(read_json(p)))+'</pre>'
    body+='<p>单例配对不能估计准确率；候选合法不代表语义正确或物理成功。需要结合图像按预先固定的标准复核。</p><p><a href="report.json">原始指标</a> · <a href="evaluation_rubric.json">预先固定的质量检查标准</a></p>'
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>观察方法精度对照</title><style>body{max-width:1200px;margin:auto;padding:24px;font:16px/1.6 system-ui}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}.images{display:flex;flex-wrap:wrap}figure{width:250px;margin:8px}img{width:100%;height:260px;object-fit:contain}</style>'+body)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'results/harness_image_processing_20261002/manifest.json')
    p.add_argument('--skill',type=Path,default=ROOT/'data/skills/experimental/visual_information.json')
    p.add_argument('--decision-id',required=True)
    p.add_argument('--model',default='claude-opus-5');p.add_argument('--ssh-host',default='company-planner')
    p.add_argument('--timeout',type=int,default=720);p.add_argument('--observation-timeout',type=int,default=360)
    p.add_argument('--order',choices=['free-first','skill-first'],default='free-first')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if not 0 < a.observation_timeout < a.timeout:p.error('Positive observation and reasoning budgets required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    case,images=load_case(read_json(a.manifest),a.output,a.decision_id)
    artifact=read_json(a.skill);write_json(a.output/'skill_snapshot.json',artifact)
    rubric={'frozen_before_calls':True,'method':'Visual review against current roots and semantic target, not agreement with baseline.',
            'checks':['Garment type and collar/hem orientation supported by current RGB',
                      'Target side and inward fold direction consistent with reference semantics',
                      'Selected current-registry marker lies on visually justified fabric region',
                      'No reference-image coordinates used for executable selection',
                      'No unsupported single-ply certainty; uncertainty distinguished from absence'],
            'limits':'No physical execution or independent annotated gold labels; do not report a numerical accuracy rate.'}
    write_json(a.output/'evaluation_rubric.json',rubric)
    report={'status':'PREPARED','arms':{},'source':case['source'],'model':a.model,
            'order':['free','skill'] if a.order=='free-first' else ['skill','free'],
            'scope':'Observation method and resulting image packet differ. Identical downstream reasoning contract.',
            'stage_budget_total_s':a.timeout,'observation_budget_s':a.observation_timeout,
            'reasoning_budget_s':a.timeout-a.observation_timeout,'max_edits':6}
    write_json(a.output/'report.json',report)
    if a.prepare_only:return 0
    for mode in report['order']:
        observer_model=RuntimeClaude(backend='remote',ssh_host=a.ssh_host,model=a.model,timeout_s=a.timeout)
        reasoning_model=RuntimeClaude(backend='remote',ssh_host=a.ssh_host,model=a.model,timeout_s=a.timeout)
        backend=ProbeBackend(ssh_host=a.ssh_host,timeout_s=a.timeout,image_tools=True)
        backend.progress_callback=lambda stage,event,duration_s=None,**kw:print(f'[observe tool] {stage}: {event}',flush=True)
        report['arms'][mode]=run_arm(case,images,artifact,a.output/mode,mode=mode,
            observer_backend=backend,observer_model=observer_model,reasoning_model=reasoning_model,
            model_name=a.model,timeout=a.timeout,observation_timeout=a.observation_timeout)
        report['status']='RUNNING';write_json(a.output/'report.json',report);render(a.output,report)
    report['status']='COMPLETED' if all(v['status']=='COMPLETED' for v in report['arms'].values()) else 'FAILED'
    if report['status']=='COMPLETED':
        free,skill=report['arms']['free'],report['arms']['skill']
        report['identical_reasoning_contract']=free['reasoning_contract_hash']==skill['reasoning_contract_hash']
        if not report['identical_reasoning_contract']:
            report['status']='INVALID_COMPARISON'
        report['elapsed_change_fraction']=skill['elapsed_s']/free['elapsed_s']-1
    write_json(a.output/'report.json',report);render(a.output,report)
    print(canonical({'output':str(a.output),'status':report['status']}),flush=True)
    return 0 if report['status']=='COMPLETED' else 1


if __name__=='__main__':raise SystemExit(main())
