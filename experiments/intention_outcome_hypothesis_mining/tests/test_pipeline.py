"""Synthetic data and fake subprocess only. Never calls API or robot."""
import copy,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import jsonschema
from experiments.intention_outcome_hypothesis_mining import api,contracts,dataset,report,run
from experiments.intention_outcome_hypothesis_mining.storage import read,save,sha

def observation(rid,with_hypothesis=True):
    corners=[{'goal':g,'wearer_side_basis':'UNKNOWN','before_state':'UNKNOWN','after_state':'UNKNOWN','change':'UNKNOWN',
              'evidence_level':'UNKNOWN','before_evidence':'不清楚','after_evidence':'不清楚','material_correspondence':'UNKNOWN',
              'alternative_explanation':'取景变化','original_expectation':'UNSPECIFIED','original_expectation_quote':None,'uncertainty':'无独立标注'} for g in contracts.GOALS]
    mismatch={'mismatch_id':rid+'-M001','type':'EXPECTED_NOT_CONFIRMED','intended_effect':'改变局部折叠',
              'observed_effect':'未确认','comparison_evidence':'图像不足','related_corner_goals':[],'worth_explaining':True,'uncertainty':'模拟'}
    hypothesis={'hypothesis_id':rid+'-H001','source_experience_id':rid,'intended_effect':'改变折叠','observed_effect':'未确认',
                'mismatch':'原预期未确认','observable_conditions':'可见叠层','hypothesized_mechanism':'相同的模拟机制',
                'predicted_effect':'可见边界变化','falsification_condition':'有效作用但无预测变化','evidence_level':'UNKNOWN',
                'uncertainty':'合成测试而非真实经验','mismatch_ids':[rid+'-M001'],'related_corner_goals':[],'categories':['UNMET_EXPECTATION']}
    return {'experience_id':rid,'intended_effect':{'intended_object':'原文对象','intended_changes':['改变局部折叠'],
            'explicit_expected_outcome':'UNKNOWN','original_quotes':['原始理由'],'unspecified_elements':['其他部位'],'uncertainty':'模拟'},
            'observed_effect':{'whole_garment_changes':['未确认'],'global_motion':'UNKNOWN','relative_deformation':'UNKNOWN',
            'new_occlusions':[],'view_confound':'UNKNOWN','execution_evidence_limits':'执行不等于抓住','effect_determinacy':'UNKNOWN','uncertainty':'模拟'},
            'corners':corners,'mismatches':[mismatch],'hypotheses':[hypothesis] if with_hypothesis else [],'no_hypothesis_reason':'' if with_hypothesis else '证据不足'}

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='hypothesis-mining-test-');self.base=Path(self.tmp.name)
        self.out=self.base/'output';self.out.mkdir();rows=[]
        for i in range(2):
            source=self.base/f'run_{i}'/'round_000';source.mkdir(parents=True)
            for k,col in [('before',(20+i,10,30)),('after',(25+i,15,40))]:
                (source/k).mkdir();Image.new('RGB',(64,48),col).save(source/k/'rgb.png')
            execution={'physical_execution':True,'status':'EXECUTED','completed_commands':2,'acquisition':'UNKNOWN'}
            save(source/'execution.json',execution);save(source/'native_plan.json',[{'kind':'gripper','data':{'action':'close'}},{'kind':'move','data':{'target_pose_mm_deg':[1,2,3,0,0,0]}}])
            save(source/'dispatch_intent.json',{'plan':{'short_reason':'原始理由','grasp_px':[10,10],'target_px':[20,20]},'observation_sha256':sha(source/'before/rgb.png')})
            rows.append({'id':f'fixture{i}','source':str(source),'run':str(source.parent),'before':str(source/'before/rgb.png'),
                         'after':str(source/'after/rgb.png'),'before_sha256':sha(source/'before/rgb.png'),'after_sha256':sha(source/'after/rgb.png'),
                         'action':{'grasp_px':[10,10],'target_px':[20,20],'lift_height_mm':50},'execution':execution})
        self.index=self.base/'index.json';save(self.index,rows)
        self.config={'codex_bin':'codex','profile':'api','model':'gpt-6-astra','effort':'high','timeout_s':1,'max_attempts':3,'retry_delay_s':0,'heartbeat_s':1}
        self.stop=threading.Event();self.log=lambda *a,**k:None
    def tearDown(self):self.tmp.cleanup()
    def records(self):return dataset.build(self.index,self.base,self.out)[0]

    def test_dry_run_no_api_and_all_pending(self):
        with patch.object(run,'run_record',side_effect=AssertionError('API must not run')):
            code=run.main(['--index',str(self.index),'--output',str(self.out),'--dry-run','--no-progress'])
        self.assertEqual(code,0);self.assertEqual(read(self.out/'summary.json')['coverage'],0)
        self.assertEqual(len(read(self.out/'unprocessed.json')),2)
        self.assertEqual(read(self.out/'hypothesis_bank.json'),[])
        self.assertFalse((self.out/'calls').exists())

    def test_complete_preserves_duplicate_hypotheses_and_resume(self):
        count=[]
        def fake(record,root,config,stop,log):
            rid=record['experience_id'];count.append(rid);o=observation(rid)
            result={'status':'COMPLETE','output':o,'latency_s':.01,'usage':{'input_tokens':1,'output_tokens':2}}
            save(root/'calls'/rid/'result.json',result);return result
        args=['--index',str(self.index),'--output',str(self.out),'--no-progress','--workers','2']
        with patch.object(run,'run_record',side_effect=fake),patch.object(run.shutil,'which',return_value='/fake/codex'):
            self.assertEqual(run.main(args),0)
            self.assertEqual(run.main(args+['--resume']),0)
        self.assertEqual(len(count),2)
        bank=read(self.out/'hypothesis_bank.json');self.assertEqual(len(bank),2)
        self.assertEqual(bank[0]['hypothesized_mechanism'],bank[1]['hypothesized_mechanism'])
        self.assertNotEqual(bank[0]['hypothesis_id'],bank[1]['hypothesis_id'])
        self.assertEqual(read(self.out/'summary.json')['coverage'],1)

    def test_bounded_retry_and_cache(self):
        r=self.records()[0];count=[]
        def attempt(record,d,config,stop,log):
            count.append(d.name);v={'status':'FAILED','retryable':True,'error':'timeout'} if len(count)==1 else {'status':'COMPLETE','output':observation(r['experience_id'])}
            save(d/'result.json',v);return v
        a=api.run_record(r,self.out,self.config,self.stop,self.log,attempt_fn=attempt)
        b=api.run_record(r,self.out,self.config,self.stop,self.log,attempt_fn=attempt)
        self.assertEqual(a['status'],'COMPLETE');self.assertEqual(b['status'],'COMPLETE');self.assertEqual(len(count),2)

    def test_exhausted_attempts_never_reset_on_resume(self):
        r=self.records()[0];count=[]
        def fail(record,d,*args):
            count.append(d);v={'status':'FAILED','retryable':True,'error':'timeout'};save(d/'result.json',v);return v
        for _ in range(2):self.assertEqual(api.run_record(r,self.out,self.config,self.stop,self.log,attempt_fn=fail)['status'],'FAILED')
        self.assertEqual(len(count),3)
        save(self.out/'protocol.json',{})
        summary,obs,bank=report.aggregate(self.out,[r],'FINISHED_WITH_UNPROCESSED')
        self.assertEqual(summary['processed_experiences'],0);self.assertEqual(summary['completed_without_hypotheses'],0)

    def test_interrupt_does_not_start_retry(self):
        r=self.records()[0];count=[]
        def interrupt(record,d,config,stop,log):
            count.append(d);stop.set();v={'status':'INTERRUPTED','retryable':True};save(d/'result.json',v);return v
        self.assertEqual(api.run_record(r,self.out,self.config,self.stop,self.log,attempt_fn=interrupt)['status'],'INTERRUPTED')
        self.assertEqual(len(count),1)

    def test_strict_corner_quotes_and_ids(self):
        r=read(self.records()[0]['payload']);o=observation(r['experience_id'])
        jsonschema.validate(o,contracts.SCHEMA);contracts.validate_semantics(o,r)
        bad=copy.deepcopy(o);bad['corners'].pop()
        with self.assertRaises(ValueError):contracts.validate_semantics(bad,r)
        bad=copy.deepcopy(o);bad['intended_effect']['original_quotes']=['After后编造的意图']
        with self.assertRaises(ValueError):contracts.validate_semantics(bad,r)
        bad=copy.deepcopy(o);bad['hypotheses'][0]['source_experience_id']='another_case'
        with self.assertRaises(ValueError):contracts.validate_semantics(bad,r)

    def test_subprocess_contract_without_running_subprocess(self):
        r=self.records()[0];captured=[]
        class Fake:
            pid=999999999;returncode=0
            def poll(self):return 0
        def fake_popen(command,**kw):
            captured.append((command,kw['stdin'].read()))
            save(Path(command[command.index('-o')+1]),observation(r['experience_id']))
            kw['stdout'].write(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':20}})+'\n');kw['stdout'].flush()
            return Fake()
        result=api.run_attempt(r,self.out/'fake_attempt',self.config,self.stop,self.log,popen=fake_popen)
        self.assertEqual(result['status'],'COMPLETE')
        command,prompt=captured[0];self.assertIn('--ephemeral',command);self.assertEqual(command[command.index('-p')+1],'api')
        self.assertIn('原始理由',prompt);self.assertNotIn('X0002',prompt)

    def test_model_tool_call_is_rejected_without_retry(self):
        r=self.records()[0]
        class Fake:
            pid=999999999;returncode=0
            def poll(self):return 0
        def fake_popen(command,**kw):
            kw['stdout'].write(json.dumps({'item':{'type':'command_execution'}})+'\n');kw['stdout'].flush();return Fake()
        result=api.run_attempt(r,self.out/'bad_attempt',self.config,self.stop,self.log,popen=fake_popen)
        self.assertEqual(result['failure_kind'],'ISOLATION_VIOLATION');self.assertFalse(result['retryable'])

    def test_timeout_stops_simulated_process(self):
        r=self.records()[0];config={**self.config,'timeout_s':.001}
        class Fake:
            pid=999999999;returncode=None
            def poll(self):return self.returncode
        proc=Fake();stopped=[]
        def halt(p):stopped.append(p);p.returncode=-15
        with patch.object(api,'stop_process',side_effect=halt):
            result=api.run_attempt(r,self.out/'timeout_attempt',config,self.stop,self.log,popen=lambda *a,**k:proc)
        self.assertEqual(result['failure_kind'],'TIMEOUT');self.assertTrue(stopped)
        self.assertTrue(result['retryable'])

    def test_empty_hypothesis_differs_from_unprocessed(self):
        records=self.records();r=records[0]
        save(self.out/'calls'/r['experience_id']/'result.json',{'status':'COMPLETE','output':observation(r['experience_id'],False)})
        save(self.out/'protocol.json',{})
        summary,obs,bank=report.aggregate(self.out,records,'RUNNING')
        self.assertEqual(summary['processed_experiences'],1)
        self.assertEqual(summary['completed_without_hypotheses'],1)
        self.assertEqual(summary['unprocessed_experiences'],1)
        self.assertIsNone(obs[1]['observation']);self.assertEqual(bank,[])

    def test_changed_frozen_config_rejected_before_api(self):
        args=['--index',str(self.index),'--output',str(self.out),'--no-progress','--dry-run']
        self.assertEqual(run.main(args),0)
        with patch.object(run,'run_record',side_effect=AssertionError('API must not run')):
            self.assertEqual(run.main(args+['--resume','--model','different']),1)

if __name__=='__main__':unittest.main()
