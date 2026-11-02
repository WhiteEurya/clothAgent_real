from run import *
folder=OUT/'native_preflight';folder.mkdir(exist_ok=True)
(folder/'base.txt').write_text(BASE)
prompt='只做启动检查：输出READY，不提出操作规则。'
cmd=['codex']+flags(folder/'base.txt')+['debug','prompt-input',prompt]
r=subprocess.run(cmd,capture_output=True,text=True,cwd=CLEAN,timeout=120)
(folder/'context_raw.json').write_text(r.stdout);(folder/'stderr.txt').write_text(r.stderr)
markers=['cloth-contour-normal-grasp','hem_sleeve_pull_rule','failure_driven_rsi','HEM_REVEAL','clothAgent_real','approved command prefixes','arm7_']
hits=[x for x in markers if x.lower() in r.stdout.lower()]
save(folder/'audit.json',{'returncode':r.returncode,'forbidden_hits':hits,'bytes':len(r.stdout),'context_sha256':sha(folder/'context_raw.json'),'provider':'openai','profile':None,'model':'gpt-6-astra','effort':'high'})
print(json.dumps(read(folder/'audit.json')),flush=True)
if r.returncode or hits:raise SystemExit('Audit did not pass')
call('native_preflight/connection',prompt,[],obj({'status':enum('READY')}))
