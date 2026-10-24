"""Independent local-morphology unfolding trial; reviewed Arm6 actions, no node matching."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import signal

from PIL import Image

from .discovery_data import save, file_hash
from .graph_policy import checked_file, evidence_file
from .local_action_memory import load_library
from .local_action_matcher import retrieve,decide,make_requests
from .arm6_replay_media import ReplayMedia
from .replay_failures import capture_cli_failure,stop_after_failure,close_cli_runtime,cli_failed


def reviewed_anchors(proposal, action, before, folder, cancel):
    from .arm6_trial_review import review_step
    anchors={k:proposal[k] for k in ('grasp_px','target_px','confidence')}
    return review_step(before.image,checked_file(action['before']),anchors,action['template'],
        action['edge_id'],dict(status='LOCAL_ACTION',confidence=proposal['confidence']),folder,
        cancel=cancel,local_action=True)


def dispatch_reviewed(runtime,before,proposal,action,folder,media):
    """Confirmation precedes depth/IK planning and every physical command."""
    from cloth_agent.gamepad_collect import TrajectoryRecorder
    from .arm6_traj_replay import prepare_motion
    media.phase('operator_review')
    anchors=reviewed_anchors(proposal,action,before,folder/'review',runtime.cancel)
    save(folder/'operator_anchors.json',anchors)
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before planning')
    media.phase('planning')
    # A bad depth stops this pilot. No camera movement or stale pixel retries.
    operations,initial=prepare_motion(runtime,before,action['template'],anchors,folder)
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before dispatch')
    save(folder/'dispatch_intent.json',dict(edge_id=action['edge_id'],before=evidence_file(before.image),
         operator_assisted=True,graph_state_matching=False,goal='unfold',proposal=proposal))
    recorder=TrajectoryRecorder(runtime,folder)
    try:
        recorder.start();recorder.stage('local_action');recorder.check()
        runtime.arrival(initial);media.phase('execution')
        result=runtime._execute_operations(operations,folder)
        save(folder/'execution.json',result)
        if result['status']!='EXECUTED':raise RuntimeError('local action execution failed')
        recorder.check();media.phase('return_perception')
        recorder.stage('return_perception');runtime.open_at_home_on_return=True
        try:runtime.move_to_perception(folder/'return_perception')
        finally:runtime.open_at_home_on_return=False
        recorder.check()
    finally:recorder.close()
    recorder.check();media.phase('end_observation')
    after=runtime.capture_current(folder/'after');media.snapshot('after',after.image);media.phase(None)
    return after,result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image',type=Path,help='offline RGB-only trial, no robot connection')
    mode.add_argument('--execute',action='store_true',help='require GUI review before each Arm6 action')
    p.add_argument('--retrieval-only',action='store_true')
    p.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    p.add_argument('--output',type=Path)
    p.add_argument('--top-k',type=int,default=6)
    p.add_argument('--max-steps',type=int,default=3)
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--timeout-s',type=float,default=180.)
    p.add_argument('--retrieval-python',type=Path)
    a=p.parse_args(argv)
    if not 1<=a.max_steps<=20 or not 1<=a.top_k<=12 or a.timeout_s<=0:p.error('invalid limits')
    if a.execute and a.retrieval_only:p.error('--retrieval-only requires --image')
    lib=load_library(a.library);memory_hash=file_hash(a.library/'library.json')
    if not (a.library/'index_manifest.json').is_file():p.error('precompute local memory embeddings first')
    output=(a.output or Path('reports')/('arm6_local_policy_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PREPARING',goal='unfold',experiment='local_morphology_action',graph_state_matching=False,
        physical_actions=0,steps=[],operator_assisted=bool(a.execute),library_sha256=memory_hash,
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    runtime=requests=None;media=ReplayMedia(output,title='Arm6 local action | unfold',sequence_video=True)
    old=signal.getsignal(signal.SIGTERM)
    def stop(*_):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,stop)
    try:
        if not a.retrieval_only:requests=make_requests(output,a.model,a.timeout_s,max_calls=2*(a.max_steps+1))
        if a.image:
            image=a.image.resolve()
        else:
            from cloth_agent.gamepad_collect_runtime import CollectionRuntime
            runtime=CollectionRuntime(a.config)
            before=runtime.observe(output/'initial_observation');image=before.image
        previous=None;history=[]
        for index in range(1 if a.image else a.max_steps):
            if file_hash(a.library/'library.json')!=memory_hash:raise ValueError('action library changed during run')
            folder=output/f'step_{index:04d}';folder.mkdir()
            media.begin(index,image);media.phase('local_retrieval')
            save(output/'status.json',dict(result,status='LOCAL_RETRIEVAL',step=index))
            retrieval=retrieve(a.library,image,folder,top_k=a.top_k,python=a.retrieval_python,
                               exclude_edges=[r['edge_id'] for r in history[-2:]])
            if a.retrieval_only:result.update(status='OFFLINE_RETRIEVAL',robot_calls=0);break
            media.phase('local_action_selection_and_grounding')
            save(output/'status.json',dict(result,status='LOCAL_ACTION_DECISION',step=index))
            proposal=decide(a.library,image,retrieval,requests,folder,previous=previous,history=history)
            result['last_proposal']=proposal
            print('ARM6 LOCAL ACTION '+json.dumps(proposal,ensure_ascii=False),flush=True)
            if a.image:result.update(status='OFFLINE_PROPOSAL',robot_calls=0);break
            if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled after decision')
            if proposal['decision']!='ACT':
                result['status']='MODEL_REPORTED_UNFOLDED' if proposal['decision']=='DONE' else 'NO_APPLICABLE_LOCAL_ACTION'
                break
            eid=proposal['edge_id'];action=lib['actions'][eid];media.edge(eid)
            save(output/'status.json',dict(result,status='AWAITING_LOCAL_ACTION_REVIEW',step=index))
            source_image=image
            before,execution=dispatch_reviewed(runtime,before,proposal,action,folder,media)
            result['physical_actions']+=1;result['steps'].append(dict(edge_id=eid,execution=execution))
            previous=dict(before=str(source_image),proposal=proposal)
            history.append(dict(edge_id=eid,expected_change=proposal['expected_change']))
            image=before.image;result['last_observation']=str(image)
            save(output/'status.json',dict(result,status='ACTION_COMPLETE'))
        else:result['status']='STEP_LIMIT'
    except (Exception,KeyboardInterrupt) as exc:
        result.update(capture_cli_failure(output,exc))
        if runtime is not None:runtime.cancel.set();stop_after_failure(runtime,result)
    finally:
        signal.signal(signal.SIGTERM,old);close_cli_runtime(runtime,result)
        # Count confirmed commands even if return/capture failed afterwards.
        executed=[]
        for path in sorted(output.glob('step_*/execution.json')):
            execution=json.loads(path.read_text())
            if execution.get('status')=='EXECUTED':
                intent=json.loads((path.parent/'dispatch_intent.json').read_text())
                executed.append(dict(edge_id=intent['edge_id'],execution=execution))
        result['steps']=executed;result['physical_actions']=len(executed)
        if requests:result['model_metrics']=requests.metrics()
        try:
            video=media.finish(result['status'])
            if video:result['video']=video
            if media.data.get('sequence_video'):result['sequence_video']=media.data['sequence_video']
        except Exception as exc:result['media_error']=str(exc)
        save(output/'result.json',result);save(output/'status.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 1 if cli_failed(result) else 0


if __name__=='__main__':raise SystemExit(main())
