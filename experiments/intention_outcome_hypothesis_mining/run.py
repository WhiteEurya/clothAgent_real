#!/usr/bin/env python3
"""Full independent hypothesis-bank job. --dry-run never invokes Codex."""
from pathlib import Path
import argparse,concurrent.futures,fcntl,json,logging,shutil,signal,sys,threading,time,os
from tqdm import tqdm
try:
    from .storage import read,save,sha
    from .contracts import PROMPT,SCHEMA
    from .dataset import build
    from .api import run_record
    from .report import render,load_state
except ImportError:
    from storage import read,save,sha
    from contracts import PROMPT,SCHEMA
    from dataset import build
    from api import run_record
    from report import render,load_state

CODE=Path(__file__).resolve().parent
REPO=CODE.parents[1]

class ConsoleHandler(logging.Handler):
    def emit(self,record): tqdm.write(self.format(record),file=sys.stderr)

def logging_setup(root,debug):
    logger=logging.getLogger('hypothesis_bank')
    for handler in logger.handlers:handler.close()
    logger.handlers.clear();logger.setLevel(logging.DEBUG)
    fmt=logging.Formatter('%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S')
    console=ConsoleHandler();console.setLevel(logging.DEBUG if debug else logging.INFO);console.setFormatter(fmt)
    file=logging.FileHandler(root/'run.log');file.setLevel(logging.DEBUG);file.setFormatter(fmt)
    logger.addHandler(console);logger.addHandler(file)
    lock=threading.Lock()
    def log(level,event,**fields):
        record={'time':time.time(),'event':event,**fields}
        with lock:
            with (root/'events.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        getattr(logger,level)(event+' '+json.dumps(fields,ensure_ascii=False))
    return log

def config_from(args):
    return {'codex_bin':args.codex_bin,'profile':args.profile,'model':args.model,'effort':args.effort,
            'timeout_s':args.timeout,'max_attempts':args.max_attempts,'retry_delay_s':args.retry_delay,
            'heartbeat_s':args.heartbeat}

def prepare(args,root,config,log):
    frozen=root/'protocol.json'
    code_hashes={p.name:sha(p) for p in CODE.glob('*.py')}
    request_config={k:config[k] for k in ['profile','model','effort','timeout_s','max_attempts']}
    if frozen.exists():
        if not (args.resume or args.render_only):raise ValueError('Output exists. Use --resume to reuse checkpoints; never overwrite prior results.')
        protocol=read(frozen)
        if protocol['code_hashes']!=code_hashes:raise ValueError('Code differs from frozen protocol. Preserve old job; use a new output directory for changed code.')
        if protocol['request_config']!=request_config:raise ValueError('Request configuration differs from frozen protocol')
        if sha(root/'prompt.txt')!=protocol['prompt_sha256'] or sha(root/'schema.json')!=protocol['schema_sha256']:
            raise ValueError('Frozen prompt or schema changed')
        records=read(root/'experience_index.json')
        if sha(root/'experience_index.json')!=protocol['experience_index_sha256']:raise ValueError('Frozen experience index changed')
        if not args.render_only:
            for r in tqdm(records,desc='Verify frozen inputs',unit='pair',disable=args.no_progress):
                for key in ['before','after','payload']:
                    if sha(r[key])!=r[key+'_sha256']:raise ValueError('Frozen input changed: '+r['experience_id']+' '+key)
        return records
    if args.render_only:raise ValueError('No frozen experiment to render')
    # Preparation can be retried before a protocol exists; no model calls have occurred.
    if list((root/'calls').glob('*/attempt_*/started.json')):raise ValueError('Calls exist without protocol; manual audit required')
    progress=lambda rows,**kw:tqdm(rows,disable=args.no_progress,**kw)
    records,audit=build(args.index.resolve(),REPO,root,progress)
    if not records:raise ValueError('No valid physical Before/After records; inspect data_audit.json')
    (root/'prompt.txt').write_text(PROMPT);save(root/'schema.json',SCHEMA)
    snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for p in CODE.glob('*.py'):shutil.copyfile(p,snapshot/p.name)
    protocol={'version':'independent-hypothesis-bank-v1','created_at':time.time(),'index_source':str(args.index.resolve()),
              'index_source_sha256':sha(args.index),'experience_index_sha256':sha(root/'experience_index.json'),
              'request_config':request_config,'prompt_sha256':sha(root/'prompt.txt'),'schema_sha256':sha(root/'schema.json'),
              'code_hashes':code_hashes,'valid_records':len(records),'maximum_attempts':len(records)*config['max_attempts'],
              'data_scope':'All valid operations in the supplied frozen index; all outcomes retained, no prior discovery ranking.',
              'independence':'Each request receives only one raw experience; no other case observations, bank entries, human skills, old analysis or prior model answers.',
              'comparison_design':'Within-request intention extraction and whole-garment observation; not separately blinded calls. Quotes are checked but semantic interpretation can still be biased by After.',
              'retry_policy':'At most max_attempts identical-input fresh calls per record, including interrupted attempts. First valid result retained. All unsuccessful attempts preserved.',
              'execution_scope':'Offline only; no robots, policy imports, skills, RSI, knowledge updates or hypothesis merging.',
              'counts_definition':'Processed=valid complete output. Failed, interrupted, waiting and unstarted records remain unprocessed. UNKNOWN is not no finding.',
              'image_encoding':'Full original pixel dimensions, JPEG quality 90 uploads; original RGB snapshots retained.',
              'limits':'Model visual evidence not independent material truth. Mechanisms are unvalidated hypotheses. Coordinator implements fixed task but does not supply manipulation rules.'}
    save(frozen,protocol)
    log('info','dataset_frozen',indexed=audit['index_records'],valid=len(records),excluded=len(audit['excluded']),duplicates=len(audit['duplicate_index_entries']))
    return records

def execute(records,root,config,args,log):
    stop=threading.Event();previous={}
    def interrupt(signum,frame):
        stop.set();log('warning','stop_requested',signal=signum,note='Cancel active API processes, checkpoint completed records; resume later.')
    for sig in [signal.SIGINT,signal.SIGTERM]:previous[sig]=signal.signal(sig,interrupt)
    successful={r['experience_id'] for r in records if load_state(root,r['experience_id'])['status']=='COMPLETE'}
    pending=[r for r in records if r['experience_id'] not in successful]
    start=time.time();save(root/'execution_state.json',{'status':'RUNNING','pid':os.getpid(),'started_at':start,'workers':args.workers})
    render(root,records,'RUNNING',detail_ids=set())
    bar=tqdm(total=len(records),initial=len(successful),desc='Analyze experiences',unit='pair',disable=args.no_progress,dynamic_ncols=True)
    completed_this_run=0;failures=0;last_report=time.time();last_heartbeat=0;changed=set();futures={};iterator=iter(pending)
    def submit(pool):
        if stop.is_set():return
        try:r=next(iterator)
        except StopIteration:return
        save(root/'calls'/r['experience_id']/'state.json',{'status':'RUNNING','started_at':time.time()})
        futures[pool.submit(run_record,r,root,config,stop,log)]=r
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in range(args.workers):submit(pool)
            while futures:
                done,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    r=futures.pop(f);rid=r['experience_id']
                    try:result=f.result()
                    except Exception as e:result={'status':'FAILED','error':f'{type(e).__name__}: {e}'}
                    save(root/'calls'/rid/'state.json',result)
                    if result['status']=='COMPLETE':successful.add(rid)
                    else:failures+=1
                    completed_this_run+=1;changed.add(rid);bar.update(1)
                    bar.set_postfix(ok=len(successful),failed=failures,active=len(futures),refresh=True)
                    log('info','record_finished',experience_id=rid,status=result['status'],ok=len(successful),total=len(records),
                        hypotheses=len(result.get('output',{}).get('hypotheses',[])))
                    submit(pool)
                now=time.time()
                if now-last_heartbeat>=args.heartbeat:
                    elapsed=now-start;remaining=len(records)-len(successful)
                    eta=elapsed/completed_this_run*remaining if completed_this_run else None
                    state={'status':'STOPPING' if stop.is_set() else 'RUNNING','updated_at':now,'processed':len(successful),
                           'total':len(records),'active':[r['experience_id'] for r in futures.values()],
                           'failed_this_session':failures,'elapsed_s':elapsed,'eta_s_rough':eta}
                    save(root/'progress.json',state);log('info','progress',**state);last_heartbeat=now
                if now-last_report>=args.report_interval or (done and not futures):
                    render(root,records,'STOPPING' if stop.is_set() else 'RUNNING',detail_ids=changed)
                    changed.clear();last_report=now
    finally:
        stop.set();bar.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        states=[load_state(root,r['experience_id'])['status'] for r in records]
        final='COMPLETE' if all(s=='COMPLETE' for s in states) else ('INTERRUPTED' if completed_this_run<len(pending) or 'INTERRUPTED' in states else 'FINISHED_WITH_UNPROCESSED')
        summary=render(root,records,final,detail_ids=None)
        save(root/'execution_state.json',{'status':final,'pid':os.getpid(),'started_at':start,'finished_at':time.time()})
        save(root/'progress.json',summary);log('info','finished',status=final,processed=summary['processed_experiences'],
                                             total=len(records),hypotheses=summary['hypothesis_count'],report=str(root/'review.html'))
    return 0 if final=='COMPLETE' else 2

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=REPO/'reports/hindsight_fast_test_20261029/experience_index.json')
    parser.add_argument('--output',type=Path,default=REPO/'reports/intention_outcome_hypothesis_bank_v1')
    parser.add_argument('--profile',default='api');parser.add_argument('--model',default='gpt-6-astra')
    parser.add_argument('--effort',default='high');parser.add_argument('--codex-bin',default='codex')
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--max-attempts',type=int,default=3,help='Total attempts per record, including first and interruptions')
    parser.add_argument('--retry-delay',type=float,default=5);parser.add_argument('--heartbeat',type=float,default=15)
    parser.add_argument('--report-interval',type=float,default=30)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true',help='Audit/snapshot all data and render empty report; ZERO model calls')
    parser.add_argument('--render-only',action='store_true',help='Regenerate report from checkpoints; ZERO model calls')
    parser.add_argument('--debug',action='store_true');parser.add_argument('--no-progress',action='store_true')
    args=parser.parse_args(argv)
    if not (1<=args.workers<=8 and 1<=args.max_attempts<=5 and args.timeout>0 and args.heartbeat>=1 and args.report_interval>=1 and args.retry_delay>=0):
        parser.error('workers 1..8, max-attempts 1..5, positive timeout/intervals, nonnegative retry-delay required')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('A runner already holds this output directory; do not start a duplicate')
        log=logging_setup(root,args.debug);config=config_from(args)
        try:
            records=prepare(args,root,config,log)
            if args.dry_run or args.render_only:
                prior=read(root/'execution_state.json')['status'] if (root/'execution_state.json').exists() else 'PREPARED_NOT_RUN'
                summary=render(root,records,prior)
                log('info','no_model_calls',mode='dry-run' if args.dry_run else 'render-only',valid=len(records),report=str(root/'review.html'))
                return 0
            if not shutil.which(args.codex_bin):raise ValueError('codex executable not found; use --codex-bin')
            return execute(records,root,config,args,log)
        except Exception as e:
            log('error','fatal',error=f'{type(e).__name__}: {e}');return 1
        finally:
            logger=logging.getLogger('hypothesis_bank')
            for handler in list(logger.handlers):
                handler.flush();handler.close();logger.removeHandler(handler)

if __name__=='__main__':raise SystemExit(main())
