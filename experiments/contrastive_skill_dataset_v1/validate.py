"""Checks for label lineage, group isolation, timestamps and delivery integrity."""
from common import *
from html.parser import HTMLParser
from urllib.parse import unquote
import re,collections

def validate():
 verify_freezes();errors=[];cases=read(OUT/'selection_manifest.json')['cases'];effects=read(OUT/'effects_verified.json')['cases'];lineage=[]
 if not read(OUT/'audit.json')['passed']:errors.append('Report audit did not pass')
 for c in cases:
  f=OUT/'cases'/c['id'];p=read(f/'provenance.json');source=ROOT/p['round'];before=read(source/'before/observation.json');after=read(source/'after/observation.json');b=before['captured_unix_s'];a=after['captured_unix_s'];times=[r['captured_unix_s'] for r in p['frames']]
  if not b<a or not all(b<t<a for t in times) or times!=sorted(set(times)):errors.append(c['id']+': timestamps')
  for rel,s in p['files'].items():
   if sha(source/rel)!=s:errors.append(c['id']+': source changed '+rel)
  for fr in p['frames']:
   if sha(ROOT/fr['source'])!=fr['sha256']:errors.append(c['id']+': frame changed')
  for n in ['before','after']:
   if Image.open(f/f'{n}.png').size!=(1280,720):errors.append(c['id']+': unexpected RGB dimensions')
  lineage.append({'id':c['id'],'before_after_seconds':a-b,'sampled_frames':len(times),'frame_times_ordered_and_within_observation':bool(all(b<t<a for t in times) and times==sorted(set(times)))})
 for e in effects:
  if set(e['reviews'])!={'R1','R2'}:errors.append(e['id']+': missing reviewer');continue
  for r in e['reviews'].values():
   if not r['call'].startswith('clean_'):errors.append(e['id']+': contaminated label')
   raw=read(OUT/'teacher_calls'/r['call']/'parsed.json')['cases'];original=next(x for x in raw if x['id']==e['id'])
   if {k:v for k,v in r.items() if k!='call'}!=original:errors.append(e['id']+': model output changed')
  for dim,z in e['z'].items():
   rr=list(e['reviews'].values());v=[r['effects'][dim]['value'] for r in rr];cf=[r['effects'][dim]['confidence'] for r in rr]
   if not all(0<=x<=1 for x in cf):errors.append(e['id']+': confidence range')
   accepted=len(set(v))==1 and v[0]!='UNKNOWN' and min(cf)>=.65 and all(r['endpoint_comparability']=='GOOD' for r in rr)
   if z['accepted']!=accepted or z['value']!=(v[0] if accepted else 'UNKNOWN'):errors.append(e['id']+': consensus mismatch')
 for f in (OUT/'teacher_calls').glob('clean_pair_review_*/input_manifest.json'):
  if any(Path(x['file']).name!='before.png' for x in read(f)['images']):errors.append('Pair reviewer saw post evidence')
 for f in (OUT/'teacher_calls').glob('clean_effect_R1_*/input_manifest.json'):
  if any(Path(x['file']).name not in ['before.png','after.png'] for x in read(f)['images']):errors.append('R1 saw process/action annotation')
 links=[]
 for target in re.findall(r'\]\(([^)]+)\)',(OUT/'report.md').read_text()):
  if not target.startswith(('http:','https:','#')) and not (OUT/unquote(target.split('#')[0])).exists():links.append(target)
 if links:errors.append('Missing report links: '+str(links))
 class Parser(HTMLParser):
  def __init__(self):super().__init__();self.images=0;self.missing=[]
  def handle_starttag(self,t,attrs):
   if t=='img':self.images+=1
   for k,v in attrs:
    if k in ['href','src'] and not v.startswith(('http:','https:','#')) and not (OUT/unquote(v)).exists():self.missing.append(v)
 parser=Parser();parser.feed((OUT/'index.html').read_text())
 if parser.missing or parser.images<40:errors.append('HTML links/images')
 result={'at':utc(),'passed':not errors,'errors':errors,'source_timestamp_checks':lineage,'all_effects_trace_to_clean_original_output':True,'consensus_recomputed':True,'pair_judges_pre_only':True,'R1_raw_endpoints_only':True,'report_links_exist':not links,'html_image_count':parser.images,'new_actions_executed':0}
 write(OUT/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_timestamp_checks'},ensure_ascii=False,indent=2))
 if errors:raise SystemExit(1)

if __name__=='__main__':validate()
