"""Offline controlled latency ablations on a frozen two-stage planning case.

No robot/controller connection, no skill promotion. One scene is a screening
experiment, not a statistical proof or a physical-success evaluation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsonschema import validate
from cloth_agent.auto_exploration import validate_visual_plan_payload
from cloth_agent.planner_backend import RemoteClaudeBackend, parse_claude_json, claude_result_envelope
from cloth_agent.harness.planning_probe import operation_trace
from cloth_agent.motion_image_sources import resolve_motion_sources
from cloth_agent.visual_preparation import reasoning_sources


class EffortBackend(RemoteClaudeBackend):
    """Experiment-only CLI effort override, saved in the actual command audit."""
    effort = None
    spool_stdin = False

    def _run_streaming(self, command, prompt, timeout_s):
        extra=[]
        command=list(command)
        if self.effort:
            if self.effort not in {'low','medium','high','xhigh','max'}:
                raise ValueError('Invalid CLI effort')
            extra.extend(['--effort',self.effort])
        if self.spool_stdin and not self._context_files and not self._direct_images:
            marker='--record-event-timing timeout '
            if command[-1].count(marker)!=1:raise ValueError('Cannot locate raw-stdin wrapper')
            command[-1]=command[-1].replace(marker,'--record-event-timing --text-only timeout ',1)
            extra.extend(['--input-format','stream-json'])
        if extra:
            marker='claude -p '
            if command[-1].count(marker)!=1:raise ValueError('Cannot locate unique Claude invocation')
            command[-1]=command[-1].replace(marker,marker+' '.join(extra)+' ',1)
            if self._debug_session:
                self._debug_session.request['command']=command
                self._debug_session.request['experiment_effort']=self.effort
                self._debug_session.request['experiment_spool_stdin']=self.spool_stdin
                self._debug_session.write('request.json',self._debug_session.request)
        return super()._run_streaming(command,prompt,timeout_s)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text())


def load_case(iteration):
    tools = iteration / 'claude_image_tools'
    def stage(prefix):
        dirs = [p for p in tools.glob(prefix+'_*') if not p.name.endswith('_snapshot') and (p/'request.json').exists()]
        if len(dirs) != 1:
            raise ValueError(f'Expected one saved {prefix} call')
        d = dirs[0]
        req = read(d/'request.json')
        files = {p.name:p.read_text() for p in (d/'context').glob('*.json')}
        context = {}
        for entry in read(d/'context/manifest.json')['files']:
            context.update(json.loads(files[entry['file']]))
        return {'request':req, 'context':context, 'files':files}
    visual, motion = stage('visual_planning'), stage('pixel_motion')
    prep = next(tools.glob('image_preparation_*_snapshot'))
    manifest = read(prep/'snapshot_manifest.json')
    for f in manifest['files']:
        if hashlib.sha256((prep/f['path']).read_bytes()).hexdigest() != f['sha256']:
            raise ValueError('Snapshot hash mismatch: '+f['path'])
    images=[]
    for i, src in enumerate(visual['request']['image_paths']):
        src=Path(src)
        frozen=prep/'original_inputs'/f'image_{i}{src.suffix}' if i < len(manifest['original_inputs']) else prep/'prepared_handoff'/src.name
        if hashlib.sha256(frozen.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
            raise ValueError('Saved planner image differs from snapshot')
        images.append(frozen)
    registry=read(next(iteration.glob('planning_attempt_fold_*/**/reference_prevalidation.json')))
    from PIL import Image
    with Image.open(images[0]) as im: upright_width=im.width
    # Prevalidation stores RAW sensor pixels; planner image_0 is clockwise90.
    for r in registry['accepted']:
        raw=r['pixel_xy'];r['raw_pixel_xy']=raw
        r['pixel_xy']=[upright_width-1-raw[1],raw[0]]
    return visual, motion, images, prep, registry


ORDERED = '''Use an explicit information-driven decision procedure: establish the current requested
sleeve in the supplied collar-up evidence; identify its visible free cuff band and check eligible
R markers; compare only plausible visible candidates; choose an inboard chest landing region;
verify current-image coordinate provenance, vertical lift before transport and controlled release.
Skip a check already resolved by supplied evidence unless conflicting evidence exists. Preserve
UNKNOWN if evidence is insufficient. This procedure changes no required safety or output fields.'''
CONCISE = '''Keep final explanatory string fields concise (normally one or two factual sentences).
Retain all required output fields, uncertainty, source IDs and action parameters. Do not emit
extra narrative or duplicate supplied findings. Do not omit a necessary evidence check for speed.'''


def geometry_helper(context, registry):
    """Deterministic coordinate conversion, no semantic ranking or fixed answer."""
    from cloth_agent.image_tools_mcp import transform_point
    import numpy as np
    # The saved handoff supplies a verified current-image 90-degree rotation.
    lineage=context['prepared_visual_evidence']['coordinate_lineage']
    rot=next(v for v in lineage if v['image_id']=='image_6')
    inverse=np.linalg.inv(np.array([rot['to_parent'][:3],rot['to_parent'][3:],[0,0,1]])).flatten()[:6].tolist()
    return {'method':'Invert verified image_6 to-parent affine; convert all eligible R markers. No ranking.',
            'source':'saved pre-execution candidate registry and verified handoff',
            'candidates':[{'reference_id':r['reference_id'], 'image_0_pixel_xy':r['pixel_xy'],
                           'image_6_pixel_xy':transform_point(inverse,r['pixel_xy'])}
                          for r in registry['accepted']]}


def observed_context(run, stage, visual_plan=None):
    """Inline actual public Read results from A, excluding old decision leakage."""
    trace=Path(run)/'A_files_split'/stage/'trace'
    files={x['file']:x['topic'] for x in read(trace/'context/manifest.json')['files']}
    calls={c['tool_use_id']:c for c in operation_trace(trace) if c['tool']=='Read'}
    excerpts=[];seen=set()
    for line in (trace/'claude_stdout.txt').read_text().splitlines():
        item=json.loads(line)
        if item.get('type')!='user':continue
        for block in item.get('message',{}).get('content',[]):
            identity=block.get('tool_use_id')
            if block.get('type')!='tool_result' or identity not in calls or identity in seen:continue
            seen.add(identity);name=Path(calls[identity]['arguments']['file_path']).name
            if name not in files:continue
            topic=files[name]
            content=block.get('content')
            if topic=='visual_plan':
                if visual_plan is None:raise ValueError('Must replace historical decision')
                content=json.dumps({'visual_plan':visual_plan},ensure_ascii=False)
            excerpts.append({'topic':topic,'public_read_result':content})
    return {'inline_observed_context':excerpts,
            'scope':'Exactly the public data-file Read results observed in control A. Line prefixes are original Read output; no context files need reading. The motion visual_plan is newly generated in this arm, not a cached answer.'}


def check_outputs(visual, motion, case, images, prep, registry):
    validate_visual_plan_payload(visual, allowed_skill_names=case['context']['approved_skill_names'])
    selected=visual['selected_reference']
    if selected['camera']!='A' or selected['reference_id'] not in registry['executable_reference_ids']:
        raise ValueError('Grasp is not an eligible current Camera-A marker')
    if selected['reference_id'] in {r['reference_id'] for r in case['context'].get('rejected_references',[])}:
        raise ValueError('Grasp was previously rejected')
    from PIL import Image
    from cloth_agent.image_tools_mcp import pixel_hash
    from cloth_agent.visual_preparation import VERIFIED_DELIVERIES
    views=read(prep/'image_debug.json')['views']
    if isinstance(views,dict): views=list(views.values())
    bundle=case['context']['prepared_visual_evidence']
    ids=bundle['prepared_view_id_map']
    # Inspect handoff mapping rather than trust model-reported coordinates.
    aliases={new:next(v for v in views if v['image_id']==old) for old,new in ids.items() if old!=new}
    roots=[]
    for i,p in enumerate(images):
        with Image.open(p) as im: roots.append({'image_id':f'image_{i}','size':list(im.size),
              'rgb_sha256':pixel_hash(im),'verification':'VERIFIED','parent_image_id':None,'original_image_index':i})
    verified=reasoning_sources(roots, views, aliases)
    # Resolver expects the semantic original filename; use original paths, already hash checked.
    originals=[Path(p) for p in case['request']['image_paths']]
    resolved,trace=resolve_motion_sources(motion,originals,verified)
    held=False; released=False; lifted=False
    for action in resolved['actions']:
        name=action['name']; args=action['args']
        if name=='close_gripper': held=True; lifted=False
        elif name=='move' and held:
            if not lifted and not(args['target']=='grasp' and args['height_above_grasp_mm']>=30):
                raise ValueError('Missing initial >=30mm vertical lift')
            lifted=True
        elif name=='open_gripper' and held: held=False; released=True
    if held or not released: raise ValueError('Missing controlled release sequence')
    grasp=next(r['pixel_xy'] for r in registry['accepted'] if r['reference_id']==selected['reference_id'])
    return {'contract_valid':True,'eligible_reference':selected['reference_id'],'grasp_pixel_xy':grasp,
            'target_mappings':trace,'coordinate_and_sequence_checks':True,
            'physical_success':'NOT_TESTED','controller_IK':'NOT_TESTED_OFFLINE'}


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    v,m,images,prep,registry=load_case(args.iteration)
    write(out/'design.json',{'source_iteration':str(args.iteration),'frozen_snapshot':str(prep),
          'robot_execution':False,'same_scene':True,'model':args.model,'repeat_note':'One screening sample per arm plus repeated merged control; not statistical proof.',
          'requested_arms':getattr(args,'arms',None) or ['A_files_split','B_inline_split','C_inline_merged','D_ordered','E_geometry','F_concise','G_direct','C_repeat'],
          'requested_effort':getattr(args,'effort',None),'matched_read_source':str(getattr(args,'matched_run',None)),
          'arms':{'A_files_split':'original context files, separate selection and motion',
          'B_inline_split':'same full context inline, separate selection and motion',
          'C_inline_merged':'same full inputs, single selection+motion call',
          'D_ordered':'C + explicit decision procedure',
          'E_geometry':'D + deterministic candidate coordinate conversion',
          'F_concise':'C + concise explanations; same schema',
          'G_direct':'C + all images directly attached, no exploratory tools',
          'C_repeat':'repeat C to screen service/agent variability',
          'H_low_effort':'C with explicitly requested low effort on same model',
          'I_matched_inline':'A with actual observed Read excerpts inlined; same two decision calls'}})
    write(out/'input_hashes.json',[{'image_id':f'image_{i}','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p)} for i,p in enumerate(images)])
    backend=EffortBackend(ssh_host=args.ssh_host,timeout_s=args.timeout,record_event_timing=True)
    backend.effort=getattr(args,'effort',None)
    backend.spool_stdin=getattr(args,'spool_stdin',False)
    reports=[]
    visual_schema=v['request']['schema'];motion_schema=m['request']['schema']
    merged_schema={'type':'object','additionalProperties':False,'properties':{'visual':visual_schema,'motion':motion_schema},'required':['visual','motion']}
    system=v['request']['system_prompt']
    # Both request logs already contain the same public/semantic diagnostic instructions.
    # Use a neutral usage_stage to avoid appending a second copy.
    def call(arm,stage,prompt,schema,context,files=None,direct=False):
        folder=out/arm/stage;folder.mkdir(parents=True,exist_ok=False)
        if files is None:
            prompt=prompt.replace('Detailed evidence is in the context directory. Read its manifest, then only files relevant to your decision. Use paginated Read for long files. Mandatory task and execution constraints above still apply.',
                                  'All detailed evidence is supplied inline below. No context files need reading; mandatory task and execution constraints still apply.')
            prompt+='\nFull task context (same saved fields, inline):\n'+json.dumps(context,ensure_ascii=False)
        call_system=system
        if direct:
            call_system=system.replace('Inspect supplied RGB using view_image.', 'Inspect the directly attached RGB images in image_id order.')
            prompt+='\nAll nine images are directly attached in image_0 through image_8 order. No exploratory tools are available or needed; use the attached pixels and inline context.'
        write(folder/'input.json',{'prompt':prompt,'schema':schema,'context_files':files,'direct_images':direct,'effort':backend.effort})
        start=time.monotonic()
        response=backend.invoke(prompt=prompt,image_paths=images,schema=schema,system_prompt=call_system,
               context_files=files,image_edit_limit=0,max_turns=16,overall_timeout_s=args.timeout,
               debug_dir=folder/'trace',usage_run_dir=folder,usage_stage='latency_ablation',
               direct_images=direct,model=args.model)
        wall=time.monotonic()-start
        payload=parse_claude_json(response.stdout);validate(payload,schema)
        write(folder/'result.json',payload)
        envelope=claude_result_envelope(response.stdout)
        operations=operation_trace(folder/'trace');counts=dict(Counter(x['tool'] for x in operations))
        if any(x['is_image_edit'] for x in operations):raise ValueError('Read-only ablation edited an image')
        metrics={'stage':stage,'wall_s':wall,'transport':response.timings,'usage':envelope.get('usage'),
                 'models':envelope.get('modelUsage'),'cost_usd':envelope.get('total_cost_usd'),
                 'turns':envelope.get('num_turns'),'tool_counts':counts,'trace':str(folder/'trace')}
        write(folder/'metrics.json',metrics)
        return payload,metrics
    def unwrap(p):
        if p['reasoning_status']!='READY' or p['missing_information'] or p['result'] is None:
            raise ValueError('Model reported insufficient information: '+str(p['missing_information']))
        return p['result']
    for arm in (getattr(args,'arms',None) or ['A_files_split','B_inline_split','C_inline_merged','D_ordered','E_geometry','F_concise','G_direct','C_repeat']):
        start=time.monotonic();report={'arm':arm,'status':'RUNNING','calls':[]};write(out/arm/'report.json',report)
        print(json.dumps({'event':'arm_start','arm':arm,'time':time.strftime('%H:%M:%S') }),flush=True)
        try:
            if arm in ['A_files_split','B_inline_split','I_matched_inline']:
                vc=observed_context(args.matched_run,'selection') if arm=='I_matched_inline' else v['context']
                p,met=call(arm,'selection',v['request']['prompt'],visual_schema,vc,v['files'] if arm.startswith('A_') else None)
                report['calls'].append(met);vp=unwrap(p)
                mc=copy.deepcopy(m['context']);mc['visual_plan']=vp
                mf=copy.deepcopy(m['files'])
                for name,s in list(mf.items()):
                    if name!='manifest.json' and 'visual_plan' in json.loads(s):mf[name]=json.dumps({'visual_plan':vp},ensure_ascii=False,indent=2)
                manifest=json.loads(mf['manifest.json'])
                for item in manifest['files']: item['characters']=len(mf[item['file']])
                mf['manifest.json']=json.dumps(manifest)
                if arm=='I_matched_inline':mc=observed_context(args.matched_run,'motion',vp)
                p,met=call(arm,'motion',m['request']['prompt'],motion_schema,mc,mf if arm.startswith('A_') else None)
                report['calls'].append(met);mp=unwrap(p)
            else:
                context=copy.deepcopy(v['context'])
                context.update({k:val for k,val in m['context'].items() if k not in {'visual_plan','images'} and k not in context})
                prompt=('Complete the SAME original selection and pixel-motion tasks in ONE invocation. '
                  'Return {visual: selection wrapper, motion: motion wrapper}. The motion must use your newly selected grasp; no saved answer is supplied. '
                  'The selection-only prohibition on actions applies only inside visual.result; put actions in motion.result. '
                  'If either part lacks required evidence, report INSUFFICIENT there.\nSELECTION TASK:\n'+v['request']['prompt']+
                  '\nMOTION TASK:\n'+m['request']['prompt'])
                if arm in ['D_ordered','E_geometry']:prompt+='\n'+ORDERED
                if arm=='E_geometry':context['computed_geometry']=geometry_helper(context,registry)
                if arm=='F_concise':prompt+='\n'+CONCISE
                p,met=call(arm,'joint',prompt,merged_schema,context,direct=arm=='G_direct')
                report['calls'].append(met);vp,mp=unwrap(p['visual']),unwrap(p['motion'])
            report.update(validation=check_outputs(vp,mp,v,images,prep,registry),status='COMPLETED')
            write(out/arm/'decision.json',{'visual':vp,'motion':mp})
        except Exception as exc:
            report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        report['wall_s']=time.monotonic()-start
        write(out/arm/'report.json',report);reports.append(report);write(out/'results.json',reports)
        print(json.dumps({'event':'arm_end','arm':arm,'status':report['status'],'wall_s':report['wall_s'],'error':report.get('error')},ensure_ascii=False),flush=True)
    print('BENCHMARK_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iteration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ssh-host',default='company-planner')
    parser.add_argument('--model',default='claude-opus-5')
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--arms',nargs='+')
    parser.add_argument('--effort',choices=['low','medium','high','xhigh','max'])
    parser.add_argument('--matched-run',type=Path)
    parser.add_argument('--spool-stdin',action='store_true',help='Experiment-only: buffer inline prompt before starting CLI to avoid its stdin deadline.')
    run(parser.parse_args())
