"""Isolated, audited Codex API transport. No project-policy imports."""
import json,subprocess,time,hashlib,os,signal,fcntl
from pathlib import Path
from PIL import Image
import jsonschema

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def call(root,name,instruction,payload,images,schema):
 folder=root/'calls'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return read(folder/'result.json')
 last=None
 # One attempt only; preserve all artifacts. One-session all20 diagnostic uses a 900-second timeout.
 for attempt in range(1):
  f=folder/f'attempt_{attempt}';f.mkdir(exist_ok=True)
  if (f/'result.json').exists():
   last=read(f/'result.json')
   if last['status']=='COMPLETE':save(folder/'result.json',last);return last
   continue
  if (f/'started.json').exists():
   last={'status':'FAILED','error':'Interrupted attempt retained'};save(f/'result.json',last);continue
  work=f/'empty_work';work.mkdir(exist_ok=True)
  prompt=instruction+'\n\n'+json.dumps(payload,ensure_ascii=False)+'\n只处理所附输入。禁止工具调用、文件读取、网页访问。不要读取项目指令。'
  (f/'input.txt').write_text(prompt);save(f/'schema.json',schema)
  uploads=[]
  for i,p in enumerate(images):
   q=f/f'image_{i:03d}.jpg';Image.open(p).convert('RGB').save(q,quality=90);uploads.append(q)
  save(f/'manifest.json',{'images':[{'source':str(p),'sha256':sha(p),'upload':str(q),'upload_sha256':sha(q)} for p,q in zip(images,uploads)],'history_included':False,'tools_allowed':False,'prompt_sha256':sha(f/'input.txt')})
  cmd=['codex','exec','-p','api','--ephemeral','--json','-s','read-only','--skip-git-repo-check','-C',str(work),'-c','model="gpt-6-astra"','-c','model_reasoning_effort="high"','--output-schema',str(f/'schema.json'),'-o',str(f/'output.json')]
  for p in uploads:cmd+=['-i',str(p)]
  cmd+=['-'];start=time.time()
  with (root/'call_budget.lock').open('a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   used=len(list((root/'calls').glob('*/attempt_*/started.json')))
   if used>=20:raise RuntimeError('Hard model call budget 20 reached')
   save(f/'budget_admission.json',{'number':used+1,'limit':20})
   save(f/'started.json',{'time':start,'command':cmd})
  save(f/'started.json',{'time':start,'command':cmd});last={'status':'FAILED'}
  try:
   with (f/'events.jsonl').open('w') as out,(f/'stderr.log').open('w') as err:
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,text=True,cwd=work,start_new_session=True)
    try:proc.communicate(prompt,timeout=900)
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15);raise
   if proc.returncode:raise RuntimeError('codex exit '+str(proc.returncode))
   events=[]
   for l in (f/'events.jsonl').read_text().splitlines():
    try:events.append(json.loads(l))
    except ValueError:pass
   if any(e.get('item',{}).get('type') in ['command_execution','mcp_tool_call','web_search','file_change'] for e in events):raise RuntimeError('Tool isolation violation')
   x=read(f/'output.json');jsonschema.validate(x,schema)
   last={'status':'COMPLETE','output':x,'usage':next((e['usage'] for e in reversed(events) if e.get('usage')),None),'tool_calls':0}
  except Exception as e:last.update(error=f'{type(e).__name__}: {e}')
  last.update(latency_s=time.time()-start,attempt=attempt);save(f/'result.json',last)
  if last['status']=='COMPLETE':break
 save(folder/'result.json',last);return last
