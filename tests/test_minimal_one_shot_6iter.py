"""Offline contract tests; never construct a real robot runtime or call GPT."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from cloth_agent.experiments import minimal_one_shot_6iter as minimal


class MinimalOneShotTests(unittest.TestCase):
    def test_dry_run_and_help_do_not_enter_runtime(self):
        with patch.object(minimal, 'minimal_runner', side_effect=AssertionError('hardware path')):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(minimal.main(['--dry-run']), 0)
            contract = json.loads(out.getvalue())
            self.assertEqual(contract['max_iterations'], 12)
            self.assertEqual(contract['planner_calls_per_iteration'], 1)
            self.assertFalse(contract['hardware_connected'])
            self.assertEqual(contract['live_camera_views'], ['perception'])
            self.assertFalse(contract['side_view_motion'])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(minimal.main([]), 0)

    def test_actual_payload_minimal_and_reset_host_unchanged(self):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        wire = episode.planner
        originals = (wire.ROTATING_MULTIVIEW_PROMPT, wire.ROTATION_SCHEMA, episode.PreActionGate)
        initializer, host = episode.initialize, episode.prepare_action
        captured=[]
        def fake_http(folder, body, headers, timeout):
            captured.append(body)
            return dict(decision='DONE',grasp_px=None,target_px=None,motion_type=None,
                        grasp_angle_deg=None,short_reason='已展开'), {'wall_latency_s':0.}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'goal').mkdir();folder=root/'round_000';folder.mkdir()
            image = root/'rgb.png';Image.new('RGB', (8, 8)).save(image)
            Image.new('RGB',(8,8)).save(root/'goal/target.png')
            before=SimpleNamespace(image=image)
            with patch.object(episode, 'EpisodeCollectionRuntime', side_effect=AssertionError('hardware')), \
                 patch.object(episode, 'direct_once', fake_http):
                with minimal.minimal_runner():
                    (root/'prompt.txt').write_text(minimal.PROMPT)
                    episode.save(root/'protocol.json', {'gate':{}, 'schema':{}})
                    episode.plan_round(root,folder,before,{},1)
                    body=captured[0]
                    self.assertEqual(body['instructions'], minimal.PROMPT)
                    self.assertEqual(body['tools'], [])
                    self.assertEqual(len(body['input']), 1)
                    self.assertEqual(sum(c['type']=='input_image' for c in body['input'][0]['content']), 2)
                    self.assertNotIn('previous_response_id', body)
                    self.assertNotIn('right view',json.dumps(body))
                    self.assertIs(episode.initialize, initializer)
                    self.assertIs(episode.prepare_action, host)
                    recorded = json.loads((root/'protocol.json').read_text())
                    self.assertEqual(recorded['prompt'], body['instructions'])
                    self.assertEqual(recorded['observation_views'], ['perception'])
                    self.assertEqual(recorded['gate']['mode'], 'DISABLED')
                    self.assertEqual(recorded['schema'], body['text']['format']['schema'])
                    self.assertTrue((folder/'request_content_audit.json').exists())
                    # Adding any extra prompt is rejected BEFORE reaching the HTTP transport.
                    body['instructions'] += '加入额外Skill'
                    with self.assertRaises(ValueError):episode.direct_once(folder,body,{},1)
                    self.assertEqual(len(captured),1)
        self.assertEqual(wire.ROTATING_MULTIVIEW_PROMPT, originals[0])
        self.assertIs(wire.ROTATION_SCHEMA, originals[1])
        self.assertIs(episode.PreActionGate, originals[2])

    def test_real_episode_loop_never_enters_side_view_path(self):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        captures=[];requests=[]
        class FakeRuntime:
            def __init__(self, config):self.cancel=threading.Event()
            def capture_current(self, folder):
                folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
                image=folder/'rgb.png';Image.new('RGB',(32,32)).save(image)
                captures.append(str(folder))
                return SimpleNamespace(image=image,directory=folder)
            observe=capture_current
            def close(self):pass
            def stop(self):return {}
        def fake_http(folder,body,headers,timeout):
            requests.append(body)
            return dict(decision='DONE',grasp_px=None,target_px=None,motion_type=None,
                        grasp_angle_deg=None,short_reason='已展开'), {'wall_latency_s':0.}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'episode';goal=Path(tmp)/'goal.png';Image.new('RGB',(32,32)).save(goal)
            with patch.object(episode,'EpisodeCollectionRuntime',FakeRuntime), \
                 patch.object(episode.planner,'auth_headers',return_value={}), \
                 patch.object(episode,'direct_once',fake_http), \
                 patch.object(episode.one_shot_views,'view_target',side_effect=AssertionError('side view')), \
                 patch.object(episode.one_shot_views,'plan_with_right_view',side_effect=AssertionError('side motion')), \
                 patch.object(episode.one_shot_views,'RIGHT_VIEW',Path(tmp)/'MISSING_SIDE_POSE'), \
                 contextlib.redirect_stdout(io.StringIO()):
                code=minimal.main(['--execute','--no-reset','--output',str(output),'--goal-image',str(goal)])
            self.assertEqual(code,0)
            self.assertEqual(len(requests),1)
            self.assertEqual(sum(c['type']=='input_image' for c in requests[0]['input'][0]['content']),2)
            self.assertFalse((output/'right_view_pose.json').exists())
            self.assertFalse(any('right_view' in f for f in captures))
            summary=json.loads((output/'episode_summary.json').read_text())
            self.assertEqual(summary['termination_reason'],'DONE')
            self.assertTrue(summary['single_view'])
            self.assertEqual(summary['total_view_observation_s'],0.)
            self.assertNotIn('right view RGB',(output/'report.md').read_text())

    def _rejection_episode(self, error, *, persistent=False, preflight_error=None):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        requests = []
        class FakeRuntime:
            def __init__(self, config):self.cancel=threading.Event()
            def capture_current(self, folder):
                folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
                image=folder/'rgb.png';Image.new('RGB',(32,32)).save(image)
                return SimpleNamespace(image=image,directory=folder,meta={'joints_deg':{'left':[0]*7}})
            observe=capture_current
            def close(self):pass
            def stop(self):return {}
        def fake_http(folder,body,headers,timeout):
            requests.append(body)
            action=persistent or len(requests)==1
            return dict(decision='ACTION' if action else 'DONE',
                        grasp_px=[10,10] if action else None,target_px=[20,20] if action else None,
                        motion_type='lift_pull_place' if action else None,
                        grasp_angle_deg=None,short_reason='test'), {'wall_latency_s':0.}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'episode';goal=Path(tmp)/'goal.png';Image.new('RGB',(32,32)).save(goal)
            with patch.object(episode,'EpisodeCollectionRuntime',FakeRuntime), \
                 patch.object(episode.planner,'auth_headers',return_value={}), \
                 patch.object(episode,'direct_once',fake_http), \
                 patch.object(episode,'prepare_action',side_effect=error), \
                 patch.object(episode.codex_recovery,'check_stationary',side_effect=preflight_error) as guard, \
                 patch.object(episode.one_shot_views,'plan_with_right_view',side_effect=AssertionError('side motion')), \
                 contextlib.redirect_stdout(io.StringIO()):
                code=minimal.main(['--execute','--no-reset','--output',str(output),'--goal-image',str(goal)])
            summary=json.loads((output/'episode_summary.json').read_text())
            self.assertEqual(summary['physical_actions'],0)
            self.assertEqual((output/'prompt.txt').read_text(),minimal.PROMPT)
            for body in requests:
                self.assertEqual(body['instructions'],minimal.PROMPT)
                self.assertEqual(len(body['input']),1)
                self.assertEqual(len(body['input'][0]['content']),3)
                self.assertNotIn('previous_response_id',body)
            for index in range(len(requests)):
                self.assertTrue((output/f'round_{index:03d}/before/rgb.png').exists())
                self.assertTrue((output/f'round_{index:03d}/request_content_audit.json').exists())
                text=requests[index]['input'][0]['content'][0]['text']
                feedback_path=output/f'round_{index:03d}/retry_feedback.json'
                if index == 0:
                    self.assertFalse(feedback_path.exists())
                    self.assertNotIn('rejected_plan',text)
                else:
                    feedback=json.loads(feedback_path.read_text())
                    previous=output/f'round_{index-1:03d}/plan.json'
                    self.assertEqual(feedback['rejected_plan'],json.loads(previous.read_text()))
                    self.assertEqual(feedback['source_round'],f'round_{index-1:03d}')
                    self.assertFalse(feedback['physical_execution'])
                    self.assertIn(str(error),feedback['rejection_reason'])
                    self.assertTrue(text.endswith(episode.retry_feedback_text(feedback)))
                    self.assertEqual(text.count('"rejected_plan"'),1)
            return code,summary,len(requests),guard.call_count

    def test_pre_dispatch_allowlist_replans_with_exact_prompt(self):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        ik=episode.DualArmError('planned IK rejection');ik.ik_failure={}
        errors=[episode.DualArmError('metric displacement 176.39 mm outside existing 10–150 mm range'),
                episode.DepthUnavailableError('missing depth'),ik,
                episode.DualArmError('left: target outside measured workspace'),
                episode.DualArmError('table-target fallback rejected: target is outside support'),
                episode.DualArmError('table-target fallback rejected: target outside bare-table ROI'),
                episode.DualArmError('table-target fallback rejected: target pixel itself is not classified as bare table')]
        for error in errors:
            with self.subTest(error=str(error)):
                code,summary,calls,guards=self._rejection_episode(error)
                self.assertEqual((code,summary['termination_reason'],calls,guards),(0,'DONE',2,1))
                self.assertEqual(summary['same_prompt_retries_scheduled'],1)

    def test_rejections_exhaust_twelve_iterations_then_request_manual_reset(self):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        code,summary,calls,guards=self._rejection_episode(
            episode.DualArmError('metric displacement 176.39 mm outside existing 10–150 mm range'),persistent=True)
        self.assertEqual((code,summary['termination_reason'],calls,guards),(2,'ROUND_LIMIT',12,11))
        self.assertEqual(summary['total_rounds'],12)

    def test_hardware_fault_is_fatal_and_retry_preflight_must_pass(self):
        from cloth_agent.experiments import one_shot_goal_full_episode_v1 as episode
        fault=episode.DualArmError('controller fault')
        code,summary,calls,guards=self._rejection_episode(fault)
        self.assertEqual((code,summary['termination_reason'],calls,guards),(1,'SAFETY_STOP',1,0))
        code,summary,calls,guards=self._rejection_episode(
            episode.DualArmError('metric displacement 176.39 mm'),preflight_error=fault)
        self.assertEqual((code,summary['termination_reason'],calls,guards),(1,'SAFETY_STOP',1,1))

    def test_shadow_does_not_call_model_or_mutate_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            def forbidden(*args, **kwargs):raise AssertionError('extra model request')
            action = {'decision':'ACTION','grasp_px':[1,2],'target_px':[3,4]}
            gate = minimal.DisabledShadowGate(tmp, None, forbidden)
            gate.evaluate(None, None, action)
            self.assertEqual(action['grasp_px'], [1,2])
            self.assertFalse(json.loads((Path(tmp)/'gate_status.json').read_text())['model_called'])

    def _fake_episode(self, reason, no_reset):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'episode';goal=Path(tmp)/'goal.png';goal.write_bytes(b'offline stub')
            config=Path(tmp)/'config.json';config.write_text('{}')
            commands=[]
            class FakeEpisode:
                def main(self):
                    argv=minimal.sys.argv[1:];commands.append(argv)
                    self_path=Path(argv[argv.index('--output')+1]);self_path.mkdir()
                    (self_path/'episode_summary.json').write_text(json.dumps({'termination_reason':reason,'total_rounds':12 if reason=='ROUND_LIMIT' else 1}))
                    return 0
            @contextlib.contextmanager
            def fake_runner():yield FakeEpisode()
            flags=['--execute','--output',str(output),'--goal-image',str(goal),'--config',str(config)]
            if no_reset:flags.append('--no-reset')
            with patch.object(minimal,'minimal_runner',fake_runner), contextlib.redirect_stdout(io.StringIO()) as out:
                result=minimal.main(flags)
            self.assertEqual(len(commands),1)
            cmd=commands[0];self.assertEqual(cmd[cmd.index('--max-rounds')+1],'12')
            self.assertIn('--single-view',cmd)
            self.assertIn('--same-prompt-retry',cmd)
            self.assertIn('--retry-plan-feedback',cmd)
            self.assertEqual('--no-reset' in cmd,no_reset)
            self.assertNotIn('--codex-recovery',cmd)
            self.assertNotIn('--stateful-context',cmd)
            note=json.loads((output/'operator_notification.json').read_text())
            self.assertFalse(note['automatically_restarted'])
            self.assertEqual(note['manual_reset_required'],reason!='DONE')
            return result,out.getvalue()

    def test_limit_notifies_and_never_starts_second_episode(self):
        code,text=self._fake_episode('ROUND_LIMIT',True)
        self.assertEqual(code,2)
        self.assertIn('12次迭代已用完',text)
        self.assertIn('--execute --no-reset',text)

    def test_done_stops_and_default_keeps_normal_reset(self):
        code,text=self._fake_episode('DONE',False)
        self.assertEqual(code,0)
        self.assertIn('DONE',text)

    def test_early_stop_does_not_claim_twelve_rounds(self):
        code,text=self._fake_episode('SAFETY_STOP',False)
        self.assertEqual(code,1)
        self.assertIn('提前停止',text)
        self.assertNotIn('12次迭代已用完',text)

    def test_iteration_limit_cannot_be_overridden(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):minimal.main(['--dry-run','--max-rounds','7'])


if __name__ == '__main__':unittest.main()
