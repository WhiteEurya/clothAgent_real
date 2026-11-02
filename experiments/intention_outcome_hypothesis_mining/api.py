"""Audited fresh Codex subprocess per experience; bounded identical-input retries."""
import json,os,signal,subprocess,time
from pathlib import Path
from PIL import Image
import jsonschema
try:
    from .storage import read,save,sha
    from .contracts import PROMPT,SCHEMA,validate_semantics
except ImportError:
    from storage import read,save,sha
    from contracts import PROMPT,SCHEMA,validate_semantics

FORBIDDEN_TOOLS={'command_execution','mcp_tool_call','web_search','file_change'}

def decode_events(path):
    events=[]
    for line in path.read_text().splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    tools=[e for e in events if e.get('item',{}).get('type') in FORBIDDEN_TOOLS]
    if tools:raise PermissionError('Model tool isolation violation; no retry')
    return next((e.get('usage') for e in reversed(events) if e.get('usage')),None)

def stop_process(proc):
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=5)

def run_attempt(record,directory,config,stop_event,log,popen=subprocess.Popen):
    directory.mkdir(parents=True,exist_ok=True)
    work=directory/'empty_work';work.mkdir(exist_ok=True)
    payload=read(record['payload'])
    if sha(record['payload'])!=record['payload_sha256']:raise ValueError('Frozen payload changed')
    prompt=PROMPT+'\n\n'+json.dumps(payload,ensure_ascii=False)+'\n只分析本条经验；不能读取其他记录或使用工具。'
    (directory/'input.txt').write_text(prompt);save(directory/'schema.json',SCHEMA)
    images=[]
    for i,k in enumerate(['before','after']):
        source=Path(record[k])
        if sha(source)!=record[k+'_sha256']:raise ValueError('Frozen image changed')
        upload=directory/f'image_{i}.jpg'
        with Image.open(source) as im:im.convert('RGB').save(upload,quality=90)
        images.append({'source':str(source),'sha256':sha(source),'upload':str(upload),'upload_sha256':sha(upload)})
    save(directory/'manifest.json',{'experience_id':record['experience_id'],'prompt_sha256':sha(directory/'input.txt'),
                                  'schema_sha256':sha(directory/'schema.json'),'images':images,'other_experiences_in_context':False,'tools_allowed':False})
    command=[config['codex_bin'],'exec','-p',config['profile'],'--ephemeral','--json','-s','read-only','--skip-git-repo-check',
             '-C',str(work),'-c','model='+json.dumps(config['model']),'-c','model_reasoning_effort='+json.dumps(config['effort']),
             '--output-schema',str(directory/'schema.json'),'-o',str(directory/'output.json')]
    for image in images:command+=['-i',image['upload']]
    command+=['-']
    start=time.time();save(directory/'started.json',{'time':start,'command':command})
    result={'status':'FAILED','retryable':True};proc=None
    try:
        with (directory/'input.txt').open() as stdin,(directory/'events.jsonl').open('w') as stdout,(directory/'stderr.log').open('w') as stderr:
            proc=popen(command,stdin=stdin,stdout=stdout,stderr=stderr,text=True,cwd=work,start_new_session=True)
            save(directory/'process.json',{'pid':proc.pid,'start_time':start})
            next_heartbeat=start+config['heartbeat_s']
            while proc.poll() is None:
                if stop_event.wait(.25):
                    stop_process(proc);result.update(status='INTERRUPTED',error='Operator interrupted; resumable within attempt budget');break
                now=time.time()
                if now-start>=config['timeout_s']:
                    stop_process(proc);result.update(error='API timeout',failure_kind='TIMEOUT');break
                if now>=next_heartbeat:
                    log('debug','request_wait',experience_id=record['experience_id'],elapsed_s=round(now-start),attempt=directory.name)
                    next_heartbeat=now+config['heartbeat_s']
            else:
                if proc.returncode:
                    result.update(error=f'codex exit {proc.returncode}',failure_kind='PROCESS_ERROR')
                else:
                    usage=decode_events(directory/'events.jsonl')
                    result['usage']=usage
                    output=read(directory/'output.json');jsonschema.validate(output,SCHEMA);validate_semantics(output,payload)
                    result={'status':'COMPLETE','retryable':False,'output':output,'usage':usage,'tool_calls':0}
    except PermissionError as e:result.update(error=str(e),failure_kind='ISOLATION_VIOLATION',retryable=False)
    except Exception as e:result.update(error=f'{type(e).__name__}: {e}',failure_kind='OUTPUT_OR_PROCESS_ERROR')
    finally:
        if proc is not None:stop_process(proc)
    result.update(started_at=start,finished_at=time.time(),latency_s=time.time()-start)
    save(directory/'result.json',result)
    return result

def orphan_still_running(directory):
    p=directory/'process.json'
    if not p.exists():return False
    try:
        pid=read(p)['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        return str(directory) in cmd and 'codex' in cmd
    except (FileNotFoundError,ProcessLookupError):return False

def run_record(record,root,config,stop_event,log,attempt_fn=run_attempt):
    folder=root/'calls'/record['experience_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():
        cached=read(folder/'result.json')
        if cached['status']=='COMPLETE':return cached
    for n in range(1,config['max_attempts']+1):
        directory=folder/f'attempt_{n:02d}'
        if (directory/'result.json').exists():
            previous=read(directory/'result.json')
            if previous['status']=='COMPLETE':save(folder/'result.json',previous);return previous
            if not previous.get('retryable',True):save(folder/'result.json',previous);return previous
            continue
        if stop_event.is_set():return {'status':'INTERRUPTED','error':'Not started due to operator interruption'}
        if (directory/'started.json').exists():
            if orphan_still_running(directory):
                return {'status':'BLOCKED_RUNNING','error':'Previous Codex process still running; no duplicate request dispatched'}
            save(directory/'result.json',{'status':'INTERRUPTED','retryable':True,'error':'Previous process interrupted; attempt retained'})
            continue
        if n>1:
            delay=min(30,config['retry_delay_s']*(2**(n-2)))
            log('warning','retry_wait',experience_id=record['experience_id'],attempt=n,delay_s=delay)
            if stop_event.wait(delay):return {'status':'INTERRUPTED','error':'Interrupted during retry delay'}
        log('info','request_start',experience_id=record['experience_id'],attempt=n,max_attempts=config['max_attempts'])
        result=attempt_fn(record,directory,config,stop_event,log)
        log('info' if result['status']=='COMPLETE' else 'warning','request_end',experience_id=record['experience_id'],attempt=n,
            status=result['status'],latency_s=round(result.get('latency_s',0),1),error=result.get('error'))
        if result['status']=='COMPLETE' or not result.get('retryable',True):save(folder/'result.json',result);return result
        if stop_event.is_set():return result
    last=read(folder/f'attempt_{config["max_attempts"]:02d}'/'result.json')
    exhausted={**last,'status':'FAILED','error':'Attempt budget exhausted: '+last.get('error','unknown'),'attempts_exhausted':True}
    save(folder/'result.json',exhausted);return exhausted
