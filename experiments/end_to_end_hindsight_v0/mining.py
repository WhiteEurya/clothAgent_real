"""Raw-only scanner. No learned/report content is loaded."""
from pathlib import Path
import json,hashlib
from PIL import Image
from transport import read,save,sha
INTENT_KEYS=('short_reason','reason','purpose','action_purpose','expected_stop_state','expected_effect','target_region','decision')
def first(paths):return next((p for p in paths if p.is_file()),None)
def image_ok(p):
 try:
  with Image.open(p) as im:im.verify()
  return True
 except Exception:return False

def scan(repo,out):
 if (out/'experience_index.json').exists():return read(out/'experience_index.json')
 native=[]
 for base in [repo/'reports',repo/'data/demonstrations',repo/'data/experiments',repo/'data/fold_experience']:
  if base.exists():native.extend(base.rglob('native_plan.json'))
 records=[];audit=[];seen={};sources=[]
 for np in sorted(set(native)):
  if out in np.parents:continue
  folder=np.parent
  ep=first([folder/'execution.json',folder/'result.json'])
  if not ep: audit.append({'source':str(np),'reason':'NO_EXECUTION_RECORD'});continue
  try:
   ex=read(ep);ops=read(np)
   if not isinstance(ex,dict) or ex.get('physical_execution') is not True:
    audit.append({'source':str(ep),'reason':'NOT_PHYSICAL'});continue
   if ex.get('status') not in ['EXECUTED','EXECUTION_STOPPED','USER_INTERRUPTED','EXECUTION_EXCEPTION','EXECUTION_FAILED']:
    audit.append({'source':str(ep),'reason':'NO_CONFIRMED_DISPATCH'});continue
   if not isinstance(ops,list) or not any(v.get('kind')=='gripper' and v.get('data',{}).get('action')=='close' for v in ops):
    audit.append({'source':str(ep),'reason':'NO_GRASP_MANIPULATION'});continue
   contexts=[folder,folder.parent,folder.parent.parent]
   before=first([c/p for c in contexts for p in ['before/rgb.png','before_capture/observation/rgb.png','observation/rgb.png']])
   after=first([folder/'after/rgb.png',folder.parent/'after/rgb.png'])
   if not before or not after or before==after or not image_ok(before) or not image_ok(after):
    audit.append({'source':str(ep),'reason':'MISSING_OR_INVALID_BEFORE_AFTER'});continue
   actual=first([folder/'states.jsonl',folder/'controller/events.jsonl',folder.parent/'states.jsonl',folder/'gripper_004.json'])
   if not actual:
    audit.append({'source':str(ep),'reason':'NO_RAW_EXECUTION_EVIDENCE'});continue
   plan=first([c/n for c in contexts for n in ['plan.json','fast_plan.json']]);p=read(plan) if plan else {}
   if not isinstance(p,dict):p={}
   intent={k:p[k] for k in INTENT_KEYS if p.get(k) is not None};intent=intent or {'status':'UNKNOWN'}
   action={k:p.get(k) for k in ['grasp_px','target_px','grasp_angle_deg','motion_type']}
   pts=first([c/'points.json' for c in contexts]);action['points_record']=read(pts) if pts else None
   geometry=first([c/'geometry.json' for c in contexts]);action['geometry']=read(geometry) if geometry else None
   # Same image pair + actually planned native operations = duplicate, irrespective of intent edits.
   key=hashlib.sha256((sha(before)+sha(after)+json.dumps(ops,sort_keys=True)).encode()).hexdigest()
   if key in seen:
    seen[key]['duplicate_sources'].append(str(ep));continue
   ident='X'+key[:14]
   # Preserve hashes of original records including metadata, without feeding source IDs to discovery model.
   bfmeta=before.parent/'observation.json';afmeta=after.parent/'observation.json'
   bm=read(bfmeta) if bfmeta.exists() else {};am=read(afmeta) if afmeta.exists() else {}
   episode=next((q for q in [folder,*folder.parents] if q.name.startswith('episode_')),None)
   if episode is None:episode=next((q for q in folder.parents if q.name.startswith('iter_')),folder).parent
   fs=[before,after,np,ep,actual]+([plan] if plan else [])+([geometry] if geometry else [])
   rec={'id':ident,'source':str(folder.relative_to(repo)),'episode':str(episode.relative_to(repo)),'round':folder.name,'execution':ex,'before':str(before),'after':str(after),'intent':intent,'action':action,'native_plan':str(np),'actual_evidence':str(actual),'timestamp':bm.get('captured_at',bm.get('timestamp',before.stat().st_mtime)),'camera_metadata':{k:bm.get(k) for k in ['camera_id','camera_serial','calibration_id','joints_deg']},'before_metadata':str(bfmeta) if bfmeta.exists() else None,'after_metadata':str(afmeta) if afmeta.exists() else None,'raw_sha256':{str(f):sha(f) for f in fs},'duplicate_sources':[],'is_partial_execution':ex.get('status')!='EXECUTED'}
   records.append(rec);seen[key]=rec;sources+=fs
  except Exception as e:audit.append({'source':str(ep),'reason':'RAW_PARSE_ERROR','error':str(e)})
 records.sort(key=lambda x:x['id'])
 save(out/'experience_index.json',records);save(out/'source_audit.json',{'native_files_scanned':len(native),'unique_experiences':len(records),'duplicate_records':sum(len(x['duplicate_sources']) for x in records),'excluded':audit,'scope':'all native-plan roots under reports, demonstrations, experiments and fold_experience; without verifiable before/after or native execution evidence cannot form experience','no_policy_reports_read':True})
 save(out/'allowed_sources.json',{'files':{str(p):sha(p) for p in sorted(set(sources))},'only_model_inputs':'sanitized original intention/action plus raw images; source paths not sent to models'})
 return records
