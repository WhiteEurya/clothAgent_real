"""Experiment-only reset target selection and automatic execution after existing checks."""
from pathlib import Path
import json,random
from unittest.mock import patch
import numpy as np

def initialize(runtime,root,state,io,authorize_execution):
    from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
    from cloth_agent.dual_arm.observation import Observation
    from cloth_agent.preference_one_shot import verify_scene
    folder=root/'initialization'
    radius_mm=random.SystemRandom().uniform(20.,40.)
    original_request=episode.direct_once
    original_execute=runtime._execute_operations
    def audited_request(directory,body,headers,timeout):
        # Forward the full-episode request unchanged, through the saved transport.
        directory=Path(directory)
        io.save(directory/'audited_payload.json',body)
        io.save(directory/'attachments.json',[{'path':str(folder/'before/rgb.png'),
            'sha256':io.sha(folder/'before/rgb.png')}])
        io.save(directory/'context_audit.json',{'source':'one_shot_goal_full_episode_v1.initialize',
            'payload_sha256':io.digest(body),'reset_prompt_unchanged':True,
            'project_skill_injection':False,'provider_internal_context':'not inspectable'})
        return original_request(directory,body,headers,timeout)
    def checked_execute(operations,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        native=[{'kind':k,'data':v.to_dict() if k=='move' else v} for k,v in operations]
        io.save(directory/'native_plan.json',native)
        binding={'stage':'RESET_OR_POSITIONING','native_sha256':io.digest(native),'folder':str(directory)}
        is_reset=directory.name=='lift_and_drop'
        if is_reset and state['reset_dispatches']>=1:raise io.StopExperiment('reset次数上限')
        image='<img width="900" src="../grasp_preview.png">' if is_reset else ''
        (directory/'review.html').write_text('<meta charset="utf-8"><h1>Reset/准备运动记录</h1>'+image+'<pre>'+__import__('html').escape(json.dumps(binding,ensure_ascii=False,indent=2))+'</pre><p>完整轨迹：native_plan.json</p>')
        authorize_execution(directory,binding)
        if io.digest(io.read(directory/'native_plan.json'))!=binding['native_sha256']:raise io.StopExperiment('Reset规划后轨迹变化')
        if is_reset:
            obs=Observation.load(folder/'before');fresh=runtime.capture_current(folder/'pre_reset_dispatch')
            verify_scene(obs,fresh,{'grasp_px':io.read(folder/'grasp.json')['pixel_xy']},directory,automatic=False)
        if operations:
            kind,data=operations[0]
            runtime.arrival(data.start if kind=='move' else {k:np.asarray(v) for k,v in data['joints'].items()})
        io.safety_assert(runtime)
        if is_reset:state['reset_dispatches']+=1
        io.save(root/'status.json',state)
        result=original_execute(operations,directory)
        if result['status']!='EXECUTED':raise io.StopExperiment('Reset/准备执行异常')
        return result
    runtime._execute_operations=checked_execute
    try:
        with patch.object(episode,'direct_once',audited_request):
            return episode.initialize(runtime,folder,episode.planner.auth_headers(),600,center_radius_mm=radius_mm,press_after_drop=False)
    finally:runtime._execute_operations=original_execute
