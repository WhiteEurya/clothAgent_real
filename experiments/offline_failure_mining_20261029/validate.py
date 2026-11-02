from index import *
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v:self.links.append(v)
def main():
 d=read(OUT/'index.json');p=read(OUT/'protocol.json');n=0
 for r in d['runs']:
  by={s['id']:s for s in r['steps']}
  for s in r['steps']:
   assert p['start_unix']<=s['timestamp']<=p['end_unix']
   if s['valid_pair']:
    assert s['execution']['status']=='EXECUTED' and s['execution']['physical_execution']
    assert s['before'] and s['after'] and s['evaluation'];n+=1
  for threshold,screen in r['screen'].items():
   for w in screen['windows']:
    seq=[by[i] for i in w['ids']];assert len(seq)==3
    ds=[s['evaluation']['coverage']['delta_ratio'] for s in seq]
    assert ds==w['deltas'] and all(x<=float(threshold) for x in ds)
    assert w['plateau']==all(abs(x)<=float(threshold) for x in ds)
    nums=[int(s['step'].split('_')[-1]) for s in seq];assert nums==list(range(nums[0],nums[0]+3))
 assert n==d['summary']['valid_pairs']
 a=read(OUT/'independent_analysis.json');assert len(a['competing_explanations'])>=2 and a['priority_experiment']['action']
 packet=read(OUT/'analysis_packet.json');assert len(packet['groups'])==6 and len(packet['controls'])==2
 forbidden=['short_reason','action_purpose','expected_stop_state','mechanism_hypothesis','policy_code']
 def inspect(x):
  if isinstance(x,dict):
   assert not any(k in x for k in forbidden)
   for v in x.values():inspect(v)
  elif isinstance(x,list):
   for v in x:inspect(v)
 inspect(packet)
 parser=Links();parser.feed((OUT/'review.html').read_text());bad=[]
 for url in parser.links:
  path=urlsplit(url).path
  if path and not urlsplit(url).scheme and not (OUT/unquote(path)).exists():bad.append(url)
 assert not bad,bad[:10]
 result={'validated_pairs':n,'candidate_sequences':6,'control_images':2,'competing_explanations':len(a['competing_explanations']),'html_links_checked':len(parser.links),'broken_local_links':len(bad),'forbidden_planner_fields_in_packet':False,'physical_actions':0}
 save(OUT/'validation.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
