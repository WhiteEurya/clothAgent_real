"""Saved-observation A/B test at the existing remote visual-planning boundary."""
from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

from ..auto_exploration import validate_visual_plan_payload
from ..planner_backend import claude_result_envelope, parse_claude_json
from .common import canonical, digest, read_json, write_json
from .direct_information_probe import prepare_skill_views
from .information_probe import ROOT, ProbeBackend, prepare_observation, tool_metrics
from .model import RuntimeClaude
from .policy import PolicyError, validate_schema


def load_case(manifest, output, decision_id=None):
    matches = [d for d in manifest['decisions'] if d['decision_id']==decision_id] if decision_id else manifest['decisions'][-1:]
    if len(matches)!=1: raise ValueError('Select exactly one saved decision')
    trace=matches[0]; request=trace['compiler_only_request']
    obs,images,source=prepare_observation(manifest,output,trace['decision_id'])
    # Do not silently remove historical/hint roots when claiming the original input contract.
    if len(images)!=len(request['image_paths']):
        raise ValueError('Pilot supports saved requests containing only clean/overlay/reference roots')
    context_dir=Path(trace['debug_directory'])/'context'
    index=read_json(context_dir/'manifest.json'); files={'manifest.json':(context_dir/'manifest.json').read_text()}
    context={}
    for entry in index['files']:
        name=entry['file']
        if not re.fullmatch(r'[0-9]+\.json',name): raise ValueError('Unexpected context filename')
        files[name]=(context_dir/name).read_text()
        item=json.loads(files[name]); context.update(item)
    case={'observation':obs,'source':source,'fold_goal':trace['pre_decision']['fold_goal'],
          'prompt':request['prompt'],'system_prompt':request['system_prompt'],'schema':request['schema'],
          'context_files':files,'context':context,'context_hash':digest(files),
          'registry':trace['pre_decision']['candidate_registry'],
          'original_budget':{k:request.get(k) for k in ('max_turns','image_edit_limit')}}
    write_json(Path(output)/'case.json',case)
    return case,images


def validate_plan(result,case):
    validate_schema(result,case['schema'])
    validate_visual_plan_payload(result,allowed_skill_names=case['context']['approved_skill_names'])
    selection=result['selected_reference']; registry=case['registry']
    if registry.get('binding')!='RAW_RGB_HASH_VERIFIED': raise PolicyError('Unverified saved registry binding')
    match=next((r for r in registry['candidates'] if r['camera']==selection['camera'] and r['candidate_id']==selection['reference_id']),None)
    if match is None: raise PolicyError('Selected reference absent from the current registry')
    if any(r['camera']==selection['camera'] and r['reference_id']==selection['reference_id']
           for r in case['context'].get('rejected_references',[])):
        raise PolicyError('Selected a rejected reference')
    allowed=case['context'].get('locally_executable_reference_ids')
    if allowed is not None and selection['reference_id'] not in allowed:
        raise PolicyError('Selected reference not in the saved executable set')
    return {'reference_in_current_registry':True,'pixel_xy':match['pixel_xy'],
            'meaning':'Schema and saved reference validity only; not semantic or physical success.'}


def response_groups(trace_directory):
    path=Path(trace_directory)/'claude_stdout.txt'; groups={}
    if not path.exists(): return None
    for line in path.read_text().splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        if event.get('type')=='assistant':
            message=event.get('message',{}); mid=message.get('id')
            if mid: groups[mid]=True
    return len(groups)


def operation_trace(trace_directory):
    """Ordered actual tool calls, deduplicated by tool-use ID, with response groups."""
    path=Path(trace_directory)/'claude_stdout.txt'
    if not path.is_file(): return []
    groups={}; seen=set(); calls=[]
    for line in path.read_text().splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        if event.get('type')!='assistant': continue
        message=event.get('message',{});mid=message.get('id')
        if not mid: continue
        groups.setdefault(mid,len(groups)+1)
        for block in message.get('content',[]):
            if block.get('type')!='tool_use' or not block.get('id') or block['id'] in seen: continue
            seen.add(block['id'])
            tool=block.get('name','UNKNOWN')
            calls.append({'index':len(calls)+1,'response_group':groups[mid], 'tool_use_id':block['id'],
                          'tool':tool,'arguments':block.get('input',{}),
                          'is_image_edit':tool.endswith(('__crop_image','__rotate_image','__resize_image'))})
    return calls


def processing_coverage(arms):
    original=arms.get('original',{});skill=arms.get('skill_then_reasoning',{})
    edits=original.get('image_edit_calls',0);groups=original.get('image_edit_response_groups',0)
    qualified=(original.get('status')=='COMPLETED' and skill.get('status')=='COMPLETED'
               and edits>=2 and groups>=2 and skill.get('host_image_ops',0)>=2)
    return {'qualified_multi_step_comparison':qualified, 'original_image_edits':edits,
            'original_image_edit_response_groups':groups,
            'criterion':'Both complete; original performs >=2 edits in >=2 response groups; skill executes >=2 host edits.',
            'note':'Response groups are observable model/tool interaction batches, not internal thought counts.'}


def run_planning_arm(case,images,artifact,output,*,use_skill,model,backend,model_name,timeout=600,turns=16,
                     skill_flow='information',max_supplements=1):
    if skill_flow not in ('information','legacy-preprocess'):
        raise ValueError('Unknown skill flow')
    if use_skill and skill_flow=='information':
        from .information_flow import run_information_flow
        return run_information_flow(case,images,artifact,output,model=model,model_name=model_name,
                                    timeout=timeout,max_supplements=max_supplements)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report={'status':'RUNNING','with_preprocessing':use_skill,'standalone_verification_calls':0,
            'common_context_hash':case['context_hash'],'common_observation_hash':digest(case['observation']),
            'model_requested':model_name,'semantic_correctness':'NOT_INDEPENDENTLY_VERIFIED',
            'host_image_ops':0,'preprocessing_s':0,'planner_image_edit_limit':0 if use_skill else 6}
    write_json(output/'report.json',report)
    try:
        planner_images=images; prompt=case['prompt']
        if use_skill:
            tasks=[{'id':'current_fold_target','information_need':
                    '为当前折叠步骤识别相关衣片边界、可见结构及其对应的当前 Rxxx 标记；不预先选定最终抓取点。',
                    'fold_goal':case['fold_goal'],'fold_state_reference':case['context'].get('fold_state_reference')}]
            planner_images,catalog,steps,metrics=prepare_skill_views(model,case['observation'],images,artifact,
                output/'preprocess',timeout=min(180,timeout),information_tasks=tasks)
            report.update(metrics)
            # Only actual view provenance is handed to reasoning, never a separate model verdict.
            prompt+='\nAn existing image skill has already prepared the following views. Inspect useful originals '
            prompt+='and prepared views together, then make the ORIGINAL requested visual-planning decision. '
            prompt+='No separate acceptance report is needed. Do not recreate crops or resize/rotate images. '
            prompt+='If information is insufficient, explain the uncertainty within the original response contract. '
            prompt+='Rxxx identities refer to the original current overlay, including in derived views. '
            prompt+='to_original maps to the original of that role, never to a reference image.\n'
            prompt+=canonical({'prepared_image_catalog':catalog,'executed_steps':[
                {k:v for k,v in step.items() if k!='view'} for step in steps]})
        remaining=timeout-(time.monotonic()-start)
        if remaining<=0: raise TimeoutError('End-to-end budget exhausted before planning')
        write_json(output/'planner_input.json',{'prompt':prompt,'schema':case['schema'],
            'system_prompt':case['system_prompt'],'context_files':case['context_files'],
            'image_paths':[str(p) for p in planner_images]})
        print('[planning] '+('skill → original reasoning' if use_skill else 'original reasoning with image tools'),flush=True)
        t=time.monotonic()
        response=backend.invoke(prompt=prompt,image_paths=planner_images,schema=case['schema'],
            system_prompt=case['system_prompt'],context_files=case['context_files'],
            image_edit_limit=report['planner_image_edit_limit'],max_turns=turns,model=model_name,
            overall_timeout_s=remaining,debug_dir=output/'trace',usage_run_dir=output,usage_stage='planning_ab')
        report['planning_s']=time.monotonic()-t
        (output/'stdout.jsonl').write_text(response.stdout);(output/'stderr.txt').write_text(response.stderr)
        result=parse_claude_json(response.stdout);write_json(output/'result.json',result)
        report.update(timings=response.timings,models=claude_result_envelope(response.stdout).get('modelUsage'),
                      **tool_metrics(response.stdout,output/'trace'),
                      planner_assistant_response_groups=response_groups(output/'trace'))
        sequence=operation_trace(output/'trace')
        write_json(output/'operation_sequence.json',sequence)
        report['image_edit_response_groups']=len({c['response_group'] for c in sequence if c['is_image_edit']})
        if use_skill and report['image_edit_calls']:
            raise PolicyError('Planner edited images despite fixed preprocessing')
        report.update(validation=validate_plan(result,case),status='COMPLETED',
                      selected_reference=result['selected_reference'],confidence=result['confidence'])
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',
                      failure_timings=getattr(exc,'timings',{}))
    finally:
        report.update(elapsed_s=time.monotonic()-start,preprocessing_calls=list(model.calls) if use_skill else [],
                      model_invocations=(len(model.calls) if use_skill else 0)+(1 if 'planner_input.json' in {p.name for p in output.iterdir()} else 0))
        write_json(output/'report.json',report)
    return report


def write_comparison(output,report):
    output=Path(output)
    content='<h1>图像 skill 接入原视觉推理：结果与耗时</h1><p>相同 obs、原任务上下文、推理提示和输出 schema。实验组先执行固定图像流程，再进入原推理；没有单独验收调用。这里只测试视觉规划，不做机器人执行或物理成功判断。</p>'
    content+='<table><tr><th>组别</th><th>总耗时</th><th>预处理</th><th>推理</th><th>所选点</th></tr>'
    for name,a in report['arms'].items():
        content+=f'<tr><td>{html.escape(name)}</td><td>{a["elapsed_s"]:.1f}s</td><td>{a.get("preprocessing_s",0):.1f}s</td><td>{a.get("planning_s",0):.1f}s</td><td>{html.escape(a.get("selected_reference",{}).get("reference_id",a["status"]))}</td></tr>'
    content+='</table><h2>相同输入</h2><div class="grid">'
    for i in range(len(read_json(output/'obs/observation.json')['images'])):
        content+=f'<figure><a href="obs/image_{i}.png"><img src="obs/image_{i}.png"></a><figcaption>image_{i}</figcaption></figure>'
    content+='</div>'
    for name,a in report['arms'].items():
        content+=f'<h2>{html.escape(name)}</h2>'
        if a.get('mode')=='global_observe_select':
            content+='<p>全图理解同时确定缺口、选择 skill、绑定参数；宿主执行；图片与信息状态一次交给选点。没有独立绑定或验收调用。</p>'
            content+=f'<p>全图理解 {a["global_understanding_s"]:.1f}s；宿主执行 {a["host_execution_s"]:.2f}s；选点 {a["selection_s"]:.1f}s；补充观察 {a["supplements"]} 轮。</p>'
            content+='<pre>'+html.escape(json.dumps({'status':a['status'],'stop_reason':a.get('stop_reason'),
                                                    'information':a['information']},ensure_ascii=False,indent=2))+'</pre>'
        point=a.get('validation',{}).get('pixel_xy')
        if point is not None:
            clean=next(i for i in read_json(output/'obs/observation.json')['images'] if i['role']=='clean')
            w,h=clean['size'];x,y=point
            label=html.escape(a['selected_reference']['reference_id'])
            content+=(f'<p>所选 {label}，当前原图像素 ({x}, {y})；红圈仅用于结果展示。</p>'
                f'<svg viewBox="0 0 {w} {h}" style="max-height:520px;width:100%">'
                f'<image href="obs/{clean["image_id"]}.png" width="{w}" height="{h}"/>'
                f'<circle cx="{x}" cy="{y}" r="15" fill="none" stroke="#ff3636" stroke-width="4"/>'
                f'<text x="{max(10,x-45)}" y="{max(25,y-24)}" fill="#ff3636" stroke="white" stroke-width="0.5" font-size="25">{label}</text></svg>')
        path=output/name/'result.json'
        if path.exists():
            result=read_json(path)
            for key in ['selected_reference','garment_observation','opening_strategy','motion_intent','expected_observation','safety_notes']:
                content+=f'<details open><summary>{key}</summary><pre>{html.escape(json.dumps(result.get(key),ensure_ascii=False,indent=2))}</pre></details>'
        sequence=output/name/'operation_sequence.json'
        if sequence.exists():
            content+='<details><summary>实际工具顺序与响应轮次</summary><pre>'+html.escape(json.dumps(read_json(sequence),ensure_ascii=False,indent=2))+'</pre></details>'
        views=[]; debug=output/name/'trace/image_debug.json'
        if debug.exists():views=[v for v in read_json(debug)['views'] if v.get('operation')]
        execution=output/name/'preprocess/execution/execution.json'
        if execution.exists(): views += [dict(s['view'],operation=s['id']) for s in read_json(execution)['steps'] if s['status']=='EXECUTED']
        flow_execution=output/name/'observations/execution.json'
        if flow_execution.exists():
            views += [dict(v,operation=v['skill_id']) for v in read_json(flow_execution)['catalog'] if v.get('skill_id')]
        content+='<div class="grid">'
        for v in views:
            path=Path(v['path']).resolve();rel=path.relative_to(output.resolve()).as_posix()
            content+=f'<figure><a href="{html.escape(rel)}"><img src="{html.escape(rel)}"></a><figcaption>{html.escape(v["operation"])}</figcaption></figure>'
        content+='</div>'
    content+='<p>单次试验、固定顺序和缓存可能影响耗时；所选点有效不等于抓取成功，也不保证推理正确。最终须核对图片与理由。</p>'
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>视觉推理对照</title><style>body{max-width:1250px;margin:auto;padding:24px;font:16px/1.6 system-ui;background:#f5f6f8;color:#17212c}table{border-collapse:collapse}td,th{padding:10px 22px;border:1px solid #bbb}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}figure{margin:0;padding:14px;background:white}img{width:100%;height:320px;object-fit:contain;background:#111}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{padding:12px;background:white;margin:8px 0}h2{margin-top:32px}</style>'+content+'</html>')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'results/harness_image_processing_20261002/manifest.json')
    p.add_argument('--skill',type=Path,default=ROOT/'data/skills/experimental/visual_information.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--decision-id')
    p.add_argument('--model',default='claude-opus-5');p.add_argument('--ssh-host',default='company-planner')
    p.add_argument('--timeout',type=int,default=600);p.add_argument('--max-turns',type=int,default=16)
    p.add_argument('--order',choices=['original-first','skill-first'],default='original-first')
    p.add_argument('--skill-flow',choices=['information','legacy-preprocess'],default='information')
    p.add_argument('--max-supplements',type=int,choices=[0,1],default=1)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    if a.timeout<=0 or not 2<=a.max_turns<=32:p.error('Invalid time/turn budget')
    a.output.mkdir(parents=True,exist_ok=False)
    case,images=load_case(read_json(a.manifest),a.output,a.decision_id);artifact=read_json(a.skill)
    write_json(a.output/'skill_snapshot.json',artifact)
    order=['original','skill_then_reasoning'] if a.order=='original-first' else ['skill_then_reasoning','original']
    report={'status':'PREPARED','source':case['source'],'order':order,'arms':{},
            'comparison_scope':'Original saved remote visual-planning contract; excludes metric grounding and physical execution.',
            'common_context_hash':case['context_hash'],'standalone_verification_calls':0}
    report['skill_flow']=a.skill_flow
    report['max_supplements']=a.max_supplements
    write_json(a.output/'report.json',report)
    if a.prepare_only:print(canonical(report));return 0
    for arm in order:
        print('[A/B] '+arm,flush=True)
        model=RuntimeClaude(backend='remote',ssh_host=a.ssh_host,model=a.model,timeout_s=a.timeout)
        backend=ProbeBackend(ssh_host=a.ssh_host,timeout_s=a.timeout,image_tools=True)
        backend.progress_callback=lambda stage,event,duration_s=None,**kw:print(
            f'[planner] {stage}: {event}'+(f' ({duration_s:.1f}s)' if duration_s is not None else ''),flush=True)
        report['arms'][arm]=run_planning_arm(case,images,artifact,a.output/arm,use_skill=arm!='original',
            model=model,backend=backend,model_name=a.model,timeout=a.timeout,turns=a.max_turns,
            skill_flow=a.skill_flow,max_supplements=a.max_supplements)
        report['status']='RUNNING';write_json(a.output/'report.json',report)
    report['status']='COMPLETED' if all(r['status']=='COMPLETED' for r in report['arms'].values()) else 'FAILED'
    if report['status']=='FAILED' and all(r['status'] in ('COMPLETED','UNKNOWN') for r in report['arms'].values()):
        report['status']='COMPLETED_WITH_UNKNOWN'
    if report['status']=='COMPLETED':
        original=report['arms']['original'];skill=report['arms']['skill_then_reasoning']
        report['observed_time_reduction_fraction']=1-skill['elapsed_s']/original['elapsed_s']
        report['same_selected_reference']=all(original['selected_reference'][k]==skill['selected_reference'][k] for k in ('camera','reference_id'))
    report['processing_coverage']=processing_coverage(report['arms'])
    write_json(a.output/'report.json',report);write_comparison(a.output,report)
    print(canonical({'status':report['status'],'output':str(a.output)}))
    return 0 if report['status'] in ('COMPLETED','COMPLETED_WITH_UNKNOWN') else 1


if __name__=='__main__':raise SystemExit(main())
