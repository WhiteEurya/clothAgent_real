from run import *
def main():
 manifest=read(OUT/'data_manifest.json');checks=[]
 def ck(name,ok,detail=None):
  checks.append({'check':name,'pass':bool(ok),'detail':detail})
 for f,h in manifest['sources'].items():ck('source unchanged '+f,sha(DATA/f)==h)
 for version in ['initial','final','deployment']:
  fp=OUT/'skills'/f'{version}_freeze.json'
  if fp.exists():
   for g,h in read(fp)['sha256'].items():ck('skill frozen '+g+' '+version,sha(OUT/'skills'/f'{g}_{version}.json')==h)
 for f in OUT.rglob('input_manifest.json'):
  rel=str(f.parent.relative_to(OUT))
  if rel.startswith('quarantine/'):continue
  m=read(f);cmd=m['command'];ck('native provider '+rel,'-p' not in cmd and not any('dmxapi' in x for x in cmd));ck('requested model '+rel,cmd[cmd.index('-m')+1]=='gpt-6-astra' and 'model_reasoning_effort="high"' in cmd)
  ck('context gate '+rel,read(f.parent/'context_check.json')['forbidden_hits']==[])
  ck('no tool use '+rel,read(f.parent/'result.json')['tool_events']==[])
  allow=[]
  if rel.startswith('teacher_'):
   g=rel.split('/')[0][-1]
   if g!='A':
    for x in cases('train'):
     allow.append(str(DATA/x['Observation']))
     if g=='C':allow.append(str(DATA/x['NextObservation']))
  elif rel.startswith('evaluation/'):
   split=rel.split('/')[1].split('_')[0];allow=[str(DATA/x['Observation']) for x in cases(split)]
  elif rel.startswith('planner_replay/'):
   state=rel.split('/')[1].split('_')[0];allow=[str(DATA/x['Observation']) for x in cases('test') if x['id']==state]
  ck('attachment allowlist '+rel,[x['source'] for x in m['attachments']]==allow)
  for x in m['attachments']:ck('attachment hash '+rel+' '+x['file'],sha(OUT/x['file'])==x['sha256'])
  ck('prompt hash '+rel,sha(f.parent/'prompt.txt')==m['prompt_sha256'])
 for split in ['development','test']:
  fp=OUT/f'{split}_predictions_frozen.json'
  if not fp.exists():continue
  freeze=read(fp)
  for g,h in freeze['hashes'].items():ck('blind prediction preserved '+split+' '+g,sha(OUT/f'evaluation/{split}_{g}/parsed.json')==h)
  ck('prediction freeze before scoring '+split,freeze['at']<=read(OUT/f'{split}_evaluation.json')['scored_at'])
 if (OUT/'skills/final_freeze.json').exists():
  for f in (OUT/'evaluation').glob('test_*/attempt.json'):ck('final freeze before test '+f.parent.name,read(OUT/'skills/final_freeze.json')['at']<read(f)['at'])
 # Full label parity is checked only after test prediction freeze.
 if (OUT/'test_predictions_frozen.json').exists():
  effects={x['id']:x['z'] for x in read(DATA/'effects_verified.json')['cases']}
  for split in ['train','development','test']:
   for x in cases(split):ck('existing label parity '+x['id'],x['OutcomeEvidence']==effects[x['id']])
 save(OUT/'validation.json',{'at':now(),'checks':checks,'passed':sum(x['pass'] for x in checks),'failed':sum(not x['pass'] for x in checks),'limits':['context audit is configured snapshot, not packet capture','root prefreeze test exposure disclosed','schema check not proof of visual reasoning correctness']})
 print(json.dumps({'passed':sum(x['pass'] for x in checks),'failures':[x for x in checks if not x['pass']]}))
if __name__=='__main__':main()
