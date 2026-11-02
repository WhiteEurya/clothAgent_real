"""Reused isolated CLI transport only; no old policy or result content."""
from common import *

def call(name,prompt,images,schema):
 folder=OUT/'teacher_calls'/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():
  if read(folder/'result.json')['status']=='COMPLETE':return read(folder/'parsed.json')
  raise RuntimeError('Prior failed call; no silent retry '+name)
 if (folder/'attempt.json').exists():raise RuntimeError('Prior unfinished call '+name)
 if len(list((OUT/'teacher_calls').glob('*/attempt.json')))>=16:raise RuntimeError('Pilot budget exhausted')
 (folder/'prompt.txt').write_text(prompt);(folder/'base.txt').write_text(BASE);save(folder/'schema.json',schema)
 transport=[]
 for j,p in enumerate(images):
  dest=folder/'attachments'/f'{j+1:02d}.jpg';dest.parent.mkdir(exist_ok=True);Image.open(p).convert('RGB').save(dest,quality=90);transport.append(dest)
 clean=Path('/tmp/failure_driven_rsi_pilot_clean');clean.mkdir(exist_ok=True)
 cmd=['codex','exec','-p','api','--ephemeral','--skip-git-repo-check','-C',str(clean),'-s','read-only','--json','--color','never','-c','approval_policy="never"','-c',f'model_instructions_file="{folder}/base.txt"','-c','developer_instructions=""','-c','project_doc_max_bytes=0','-c','web_search="disabled"','-c',f'model_catalog_json="{OUT}/local_model_catalog.json"','-c','model_providers.dmxapi.request_max_retries=0','-c','model_providers.dmxapi.stream_max_retries=0','-c','features.skip_host_skill_discovery=true']
 for f in ['shell_tool','shell_snapshot','view_image','multi_agent','apps','plugins','hooks','memories','browser_use','computer_use','image_generation','skill_search','goals','sleep_tool','unbounded_connection_retries']:cmd+=['--disable',f]
 cmd+=['--output-schema',str(folder/'schema.json'),'-o',str(folder/'raw_output.txt')]
 for p in transport:cmd+=['-i',str(p)]
 cmd+=['-'];save(folder/'input_manifest.json',{'command':cmd,'images':[{'file':str(p.relative_to(OUT)),'sha256':sha(p)} for p in images],'transport_images':[{'file':str(p.relative_to(OUT)),'sha256':sha(p)} for p in transport],'prompt_sha256':sha(folder/'prompt.txt'),'profile_sha256':sha(Path.home()/'.codex/api.config.toml')})
 save(folder/'attempt.json',{'at':utc(),'ordinal':len(list((OUT/'teacher_calls').glob('*/attempt.json')))+1,'automatic_retry':False});print('START',name,flush=True)
 start=time.monotonic();terminal_at=None
 with (folder/'events.jsonl').open('w') as so,(folder/'stderr.txt').open('w') as se:
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=so,stderr=se,text=True,cwd=clean,start_new_session=True)
  proc.stdin.write(prompt);proc.stdin.close()
  while proc.poll() is None:
   content=(folder/'events.jsonl').read_text()
   if '"type":"turn.completed"' in content or '"type":"turn.failed"' in content:
    if terminal_at is None:terminal_at=time.monotonic()
    if (folder/'raw_output.txt').exists() or time.monotonic()-terminal_at>8:
     proc.terminate();break
   if time.monotonic()-start>480:proc.terminate();break
   time.sleep(.5)
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 events=[]
 for l in (folder/'events.jsonl').read_text().splitlines():
  try:events.append(json.loads(l))
  except:pass
 done=[e for e in events if e.get('type')=='turn.completed'];tool=[e['item'] for e in events if 'item' in e and e['item'].get('type') not in ['error','agent_message','reasoning']]
 ok=len(done)==1 and not tool and (folder/'raw_output.txt').exists();elapsed=(terminal_at or time.monotonic())-start
 result={'at':utc(),'status':'COMPLETE' if ok else 'FAILED','wall_s':elapsed,'exit_code':proc.returncode,'completed_turns':len(done),'tool_events':tool,'usage':done[0].get('usage') if done else None,'latency_note':'time to observed turn.completed; CLI shutdown excluded','model':'gpt-6-astra','effort':'high'}
 if ok:parsed=json.loads((folder/'raw_output.txt').read_text());save(folder/'parsed.json',parsed)
 save(folder/'result.json',result);freeze('call_'+name,[p for p in folder.rglob('*') if p.is_file()]);print('END',name,result['status'],round(elapsed,2),flush=True)
 if not ok:raise RuntimeError('Model request failed '+name)
 return parsed
