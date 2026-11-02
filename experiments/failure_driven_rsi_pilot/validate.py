"""Validate completed offline artifacts without making any model request."""
from common import *
import re
from urllib.parse import unquote

def main():
 verify_freezes();errors=[];checks={}
 required=['report.md','data_manifest.json','observations.json','hypotheses.json','candidate_skill.json','heldout_evaluation.json','planner_comparison/summary.json','audit.json','costs.json']
 checks['required_artifacts']=all((OUT/p).is_file() for p in required)
 checks['research_audit']=read(OUT/'audit.json')['audit_pass']
 checks['frozen_skill_is_exact_teacher_output']=read(OUT/'candidate_skill.json')==read(OUT/'teacher_calls/03_compress/parsed.json')
 checks['raw_output_preservation']=all(json.loads((f.parent/'raw_output.txt').read_text())==read(f) for f in (OUT/'teacher_calls').glob('*/parsed.json'))
 paired={}
 for cid in ['H1','H2','H3']:
  fa=OUT/'teacher_calls'/f'planner_{cid}_A';fb=OUT/'teacher_calls'/f'planner_{cid}_B'
  a=(fa/'prompt.txt').read_text();b=(fb/'prompt.txt').read_text()
  suffix='唯一新增候选规则（待验证，保持其适用条件，不强行触发）：\n'+json.dumps(read(OUT/'candidate_skill.json'),ensure_ascii=False)
  ma=read(fa/'input_manifest.json');mb=read(fb/'input_manifest.json')
  paired[cid]={'identical_base_and_only_rule_added':b==a+suffix,'identical_original_images':ma['images']==mb['images'],'identical_encoded_images':[x['sha256'] for x in ma['transport_images']]==[x['sha256'] for x in mb['transport_images']],'identical_model_base':(fa/'base.txt').read_bytes()==(fb/'base.txt').read_bytes()}
 checks['A_B_isolation']=paired
 provenance=[]
 for f in (OUT/'cases').glob('*/provenance.json'):
  d=read(f)
  for rel,s in d['source_files'].items():
   if sha(ROOT/d['round']/rel)!=s:provenance.append(str(ROOT/d['round']/rel))
  for frame in d['frames']:
   if sha(ROOT/frame['source'])!=frame['sha256']:provenance.append(frame['source'])
 checks['source_hash_errors']=provenance
 links=[]
 for target in re.findall(r'\]\(([^)]+)\)',(OUT/'report.md').read_text()):
  if target.startswith(('http:','https:','#')):continue
  p=OUT/unquote(target.split('#')[0].strip('<>'))
  if not p.exists():links.append(target)
 checks['missing_report_links']=links
 if not all(checks[k] for k in ['required_artifacts','research_audit','frozen_skill_is_exact_teacher_output','raw_output_preservation']):errors.append('Required output/audit checks failed')
 if not all(all(d.values()) for d in paired.values()):errors.append('A/B input mismatch')
 if provenance or links:errors.append('Source hash or report link failure')
 result={'at':utc(),'passed':not errors,'checks':checks,'errors':errors,'scope':'Artifact integrity, leakage, paired inputs and source provenance; no physical validity claim.'}
 (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if errors:raise SystemExit(1)

if __name__=='__main__':main()
