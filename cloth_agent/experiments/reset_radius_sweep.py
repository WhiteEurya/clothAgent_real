"""Three valid contextual unfolding trials per reset radius, 20 through 80 mm."""
import argparse
import csv
import html
import math
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

from . import automatic_top5_batch as batch
from . import one_shot_goal_continuous_v1 as base
from cloth_agent import unfold_experiment_policy as policy

RADII = tuple(range(20,81,10))


class Control(batch.Control):
    page=batch.PAGE.replace('自动展开 · 3次采集','Reset 半径递增实验').replace(
        '执行GPT选择，不使用学习参数。',
        'd=20、30…80 mm，每档3次有效试验；0/3时结束。精简规划不评价上步效果；最多12个动作。')
    page=page.replace('每次结束后，手动整理并确认才开始下一次随机初始化。','DONE且清理完成后自动开始下一次随机初始化；失败或异常后等待手动整理并确认。')
    page=page.replace('第 ${state.episode} / ${state.episodes_limit} 次','第 ${state.episode} 次尝试（含补跑）')
    page=page.replace('`状态：${phases[state.phase]||state.phase}`,',
        '`状态：${phases[state.phase]||state.phase}`,`当前 d=${state.d_mm} mm，本档有效 ${state.radius_result.valid_trials}/3，成功 ${state.radius_result.successes} 次`,')
    def __init__(self,root):
        # Room for invalid trials; scientific limit is 3 valid trials per radius.
        super().__init__(root,9999,True)
        self.d_mm=20;self.stop_reason=None

    def status(self):
        state=super().status()
        with self.lock:
            state.update(d_mm=self.d_mm,radius_result=policy.radius_summary(self.records,self.d_mm),
                         experiment_stop_reason=self.stop_reason)
        if (self.root/'index.html').exists():
            state['reports'].insert(0,dict(label='各半径成功率与耗时汇总',url='/files/index.html'))
        return state


def report(root, records, stop_reason):
    groups=[policy.radius_summary(records,d) for d in RADII if any(r['d_mm']==d for r in records)]
    base.save(root/'sweep_results.json',dict(updated_at=base.utc(),stop_reason=stop_reason,
        success_definition='model DONE; not independently verified physical success',
        radii=groups,trials=records))
    with (root/'trials.csv').open('w') as f:
        fields=['episode','d_mm','outcome','termination','actions','rounds','elapsed_s','planning_s','robot_s','gate_s','reset_seed','actual_reset_offset_mm','operator_stopped_current','detail']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    with (root/'success_rates.csv').open('w') as f:
        fields=['d_mm','attempts','valid_trials','successes','failures','invalid_trials','success_rate','complete','zero_success_stop']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(groups)
    lines=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10">',
        '<title>Reset 半径实验</title><style>body{font:18px system-ui;margin:32px}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}a{color:#1766b1}</style>',
        '<h1>Reset 半径实验</h1><p>成功=模型 DONE；失败=12个动作后仍未DONE；已停用连续无改善判断。人工中断、异常和ABSTAIN单独记录，不进入成功率分母。</p>',
        '<p>每次重新随机初始化，沿用可见衣袖/下摆 + 上一动作上下文；不使用TopN或学习参数。每档必须完成3次有效试验，才判断是否0/3。</p>',
        '<p>实验状态：'+html.escape(stop_reason or '进行中')+'</p>',
        '<table><tr><th>d (mm)</th><th>成功/有效</th><th>成功率</th><th>无效</th><th>本档完成</th></tr>']
    for g in groups:
        pct='—' if g['success_rate'] is None else f"{g['success_rate']:.1%}"
        lines.append(f"<tr><td>{g['d_mm']}</td><td>{g['successes']}/{g['valid_trials']}</td><td>{pct}</td><td>{g['invalid_trials']}</td><td>{g['complete']}</td></tr>")
    lines.append('</table><h2>各次试验</h2><table><tr><th>次序</th><th>d</th><th>结果</th><th>结束原因</th><th>动作数</th><th>总秒数</th><th>理由与可视化</th></tr>')
    for r in records:
        ep=f"episode_{r['episode']:04d}"
        links=' '.join(f'<a href="{ep}/{p.parent.name}/decision.html">{p.parent.name}</a>' for p in sorted((root/ep).glob('round_*/decision.html')))
        lines.append(f"<tr><td>{r['episode']}</td><td>{r['d_mm']}</td><td>{r['outcome']}</td><td>{html.escape(r['termination'])}</td><td>{r['actions']}</td><td>{r['elapsed_s'] or 0:.1f}</td><td>{links}</td></tr>")
    lines.append('</table>');(root/'index.html').write_text('\n'.join(lines))


def build_command(args,folder,goal,radius):
    return [sys.executable,'-u','-B','-m','cloth_agent.experiments.one_shot_goal_full_episode_v1',
        '--execute','--visible-sleeve-hem','--stateful-context','--output',str(folder),
        '--goal-image',str(goal),'--config',str(args.config.resolve()),
        '--initialize-center-radius-mm',str(radius),'--no-improvement-patience','0',
        '--uncertain-is-no-improvement','--max-actions','12','--max-rounds','13']


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    control=Control(root);server=batch.serve(control,args.port)
    (root/'goal').mkdir();shutil.copyfile(args.goal_image,root/'goal/target.png')
    files=base.SOURCE_FILES+['cloth_agent/geometry_report.py','cloth_agent/one_shot_depth_recovery.py','cloth_agent/experiments/arm6_depth_recovery.py']+['cloth_agent/experiments/automatic_top5_batch.py',
        'cloth_agent/experiments/reset_radius_sweep.py','cloth_agent/preference_one_shot.py',
        'cloth_agent/action_preference.py','cloth_agent/restricted_context_one_shot.py',
        'cloth_agent/unfold_experiment_policy.py']
    hashes={str(base.ROOT/p):base.digest(base.ROOT/p) for p in files}
    hashes[str(args.config.resolve())]=base.digest(args.config)
    hashes[str(root/'goal/target.png')]=base.digest(root/'goal/target.png')
    base.save(root/'protocol.json',dict(created_at=base.utc(),radii_mm=RADII,valid_trials_per_radius=3,
        stop_on_zero_of_three=True,max_actions=12,final_action_assessment=True,
        no_improvement_patience=0,no_improvement_verdicts=[],
        previous_effect_enabled=False,planner_output='compact_action',
        success='model DONE',invalid='operator stop, hardware/model errors, ABSTAIN; retry after manual ready',
        grasp_restriction='visible sleeves or bottom hem',context='previous executed action and before image',
        topn=False,learning=False,right_view=False,press_after_drop=False,
        radius_definition='maximum measured XY offset from detected cloth center; also constrained by detected visible cloth region',
        max_consecutive_invalid=args.max_consecutive_invalid,source_sha256=hashes,
        between_trials='automatic reset after clean SUCCESS; manual ready after FAILURE/INVALID',control_url=f'http://127.0.0.1:{args.port}'))
    source_dir=root/'source_snapshot';source_dir.mkdir()
    for path in files:
        dest=source_dir/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base.ROOT/path,dest)
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in handlers:signal.signal(s,control.request_stop)
    def checkpoint():
        report(root,control.records,control.stop_reason)
        state=control.status();state.pop('token',None);base.save(root/'batch_summary.json',state)
    def ready():
        control.transition('WAITING_READY');checkpoint()
        print(f'WAITING_READY d={control.d_mm} mm: 整理衣物并确认后开始随机初始化。',flush=True)
        while not control.ready.wait(.5):
            if control.cancel.is_set():return False
        return not control.cancel.is_set()
    print(f'CONTROL http://127.0.0.1:{args.port} OUTPUT {root}',flush=True)
    attempts=0;invalid_streak=0
    try:
        for radius in RADII:
            control.d_mm=radius
            while policy.radius_summary(control.records,radius)['valid_trials']<3:
                if control.cancel.is_set():break
                needs_ready=(args.wait_before_first if not attempts else control.records[-1]['outcome']!='SUCCESS')
                if needs_ready and not ready():break
                if attempts and not needs_ready:
                    control.event('AUTO_NEXT_AFTER_DONE',previous_episode=attempts,next_d_mm=radius)
                    print(f'AUTO_NEXT_AFTER_DONE d={radius} mm: 清理完成，自动开始下一次随机初始化。',flush=True)
                if any(base.digest(p)!=v for p,v in hashes.items()):
                    raise RuntimeError('experiment sources/input changed; no next dispatch')
                attempts+=1;control.episode_stop=False;control.stop_signal=None
                control.transition('RUNNING',attempts);checkpoint()
                folder=root/f'episode_{attempts:04d}'
                # Handles a stop arriving after RUNNING publication but before child creation.
                def watch_stop():
                    while control.phase in ('RUNNING','STOPPING') and control.child is None and not control.cancel.is_set():time.sleep(.02)
                    if control.episode_stop:control.forward_stop()
                threading.Thread(target=watch_stop,daemon=True).start()
                code=base.launch_episode(build_command(args,folder,root/'goal/target.png',radius),root/f'{folder.name}.run.log',control)
                f=folder/'episode_summary.json';record=base.json.loads(f.read_text()) if f.exists() else {}
                outcome=policy.classify(record,code,control.episode_stop or control.cancel.is_set())
                control.last_stop=record.get('termination_reason') or 'MISSING_SUMMARY'
                init_file=folder/'initialization/result.json'
                init=base.json.loads(init_file.read_text()) if init_file.exists() else {}
                grasp_file=folder/'initialization/grasp.json'
                grasp=base.json.loads(grasp_file.read_text()) if grasp_file.exists() else {}
                offset=(math.dist(grasp['world_xyz_mm'][:2],grasp['center_world_xyz_mm'][:2])
                        if 'world_xyz_mm' in grasp and 'center_world_xyz_mm' in grasp else None)
                control.records.append(dict(episode=attempts,d_mm=radius,outcome=outcome,
                    termination=control.last_stop,detail=record.get('termination_detail',''),
                    actions=record.get('physical_actions',0),rounds=record.get('total_rounds',0),
                    elapsed_s=record.get('episode_wall_s'),planning_s=record.get('total_planning_s'),
                    robot_s=record.get('total_robot_execution_s'),gate_s=record.get('total_gate_wall_s'),
                    returncode=code,operator_stopped_current=control.episode_stop,cleanup_error=record.get('cleanup_error'),
                    reset_seed=init.get('seed'),actual_reset_offset_mm=offset,human_success=None))
                invalid_streak=invalid_streak+1 if outcome=='INVALID' else 0
                checkpoint()
                if not record.get('finished_at') or record.get('cleanup_error'):
                    control.stop_reason='CLEANUP_UNCONFIRMED';break
                if invalid_streak>=args.max_consecutive_invalid:
                    control.stop_reason='REPEATED_INVALID_TRIALS';break
            if control.cancel.is_set():control.stop_reason='USER_STOP';break
            if control.stop_reason:break
            result=policy.radius_summary(control.records,radius)
            if result['zero_success_stop']:
                control.stop_reason=f'ZERO_SUCCESS_AT_{radius}_MM';break
        else:control.stop_reason='RADIUS_LIMIT_80_MM_NO_ZERO_SUCCESS'
    except BaseException as exc:
        control.request_stop(signal.SIGINT);control.stop_reason=f'{type(exc).__name__}: {exc}'
        base.save(root/'batch_error.json',dict(error=control.stop_reason));raise
    finally:
        control.transition('FINISHED');checkpoint()
        for s,h in handlers.items():signal.signal(s,h)
        server.shutdown();server.server_close()
        print(f'SWEEP FINISHED {control.stop_reason} {root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=base.ROOT/'config/dual_arm.local.json')
    p.add_argument('--goal-image',type=Path,default=base.DEFAULT_GOAL)
    p.add_argument('--port',type=int,default=8775)
    p.add_argument('--wait-before-first',action='store_true')
    p.add_argument('--max-consecutive-invalid',type=int,default=3)
    a=p.parse_args()
    if a.output.exists() or not a.goal_image.is_file() or not a.config.is_file():p.error('invalid output/goal/config')
    if not 1024<=a.port<=65535 or a.max_consecutive_invalid<1:p.error('invalid limits')
    run(a)

if __name__=='__main__':main()
