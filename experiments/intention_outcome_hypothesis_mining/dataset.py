"""Audit raw indexed operations, snapshot inputs; no prior analysis/skill reads."""
from pathlib import Path
import shutil
from PIL import Image
try:
    from .storage import read,save,sha
except ImportError:
    from storage import read,save,sha

INTENTION_FIELDS=['short_reason','reason','purpose','action_purpose','expected_effect','expected_outcome',
                  'expected_stop_state','expected_stop_condition','target_region']

def build(index_path,repo,root,progress=lambda x,**kw:x):
    index=read(index_path);records=[];exclusions=[];duplicates=[];seen=set()
    for order,old in enumerate(progress(index,desc='Audit raw records',unit='pair'),1):
        rid=f'X{order:04d}'
        try:
            source=(repo/old['source']).resolve()
            before=(repo/old['before']).resolve();after=(repo/old['after']).resolve()
            execution_path=source/'execution.json';native_path=source/'native_plan.json'
            execution=read(execution_path);native=read(native_path)
            if execution.get('physical_execution') is not True:raise ValueError('No verified physical execution flag')
            if not any(x.get('kind')=='gripper' and x.get('data',{}).get('action')=='close' for x in native):
                raise ValueError('No grasp operation in recorded native plan')
            for image in [before,after]:
                with Image.open(image) as im: im.verify()
            bh,ah=sha(before),sha(after)
            if bh!=old['before_sha256'] or ah!=old['after_sha256']:
                raise ValueError('Image hash differs from source index')
            duplicate_key=(str(source),bh,ah,sha(native_path))
            if duplicate_key in seen:
                duplicates.append({'experience_id':rid,'source':str(source),'reason':'Exact duplicate indexed operation'});continue
            seen.add(duplicate_key)
            dispatch_path=source/'dispatch_intent.json'
            dispatch=read(dispatch_path) if dispatch_path.exists() else {}
            plan=dispatch.get('plan',{})
            intention={k:plan[k] for k in INTENTION_FIELDS if plan.get(k) is not None}
            warnings=[]
            dispatch_hash=dispatch.get('observation_sha256')
            if dispatch_hash and dispatch_hash!=bh:warnings.append('Dispatch observation hash differs from indexed Before; association requires caution')
            if not intention:warnings.append('Original pre-action intention unavailable; do not infer it from After')
            compact=[]
            keep=['arm','action','target_pose_mm_deg','interpolation','cartesian_speed_mm_s',
                  'cartesian_acceleration_mm_s2','opening_delta_pulse','settle_s','target','position']
            for i,op in enumerate(native):
                compact.append({'command_index':i,'kind':op.get('kind'),'data':{k:v for k,v in op.get('data',{}).items() if k in keep}})
            directory=root/'inputs'/rid;directory.mkdir(parents=True,exist_ok=True)
            for path,name in [(before,'before.png'),(after,'after.png')]:shutil.copyfile(path,directory/name)
            hashes={str(p):sha(p) for p in [before,after,execution_path,native_path]}
            if dispatch_path.exists():hashes[str(dispatch_path)]=sha(dispatch_path)
            with Image.open(before) as im:image_size=list(im.size)
            payload={'experience_id':rid,'original_intention':intention,
                     'intention_provenance':'Original dispatch_intent plan, saved before dispatch by producer; independent timestamp proof not assumed.' if intention else 'UNKNOWN',
                     'original_dispatched_parameters':{k:v for k,v in plan.items() if k not in INTENTION_FIELDS},
                     'indexed_executed_action':old['action'],'native_command_sequence':compact,
                     'execution_information':execution,'association_warnings':warnings,
                     'image_order':['Before','After'],'image_size':image_size}
            save(directory/'payload.json',payload)
            record={'experience_id':rid,'source_index_id':old.get('id'),'source':str(source),'run':old.get('run'),
                    'captured_at':old.get('captured_at'),'original_before':str(before),'original_after':str(after),
                    'before':str(directory/'before.png'),'after':str(directory/'after.png'),'before_sha256':bh,'after_sha256':ah,
                    'payload':str(directory/'payload.json'),'payload_sha256':sha(directory/'payload.json'),
                    'raw_hashes':hashes,'original_intention':intention,'association_warnings':warnings}
            records.append(record)
        except Exception as e:
            exclusions.append({'experience_id':rid,'source_index_id':old.get('id'),'source':old.get('source'),
                               'reason':f'{type(e).__name__}: {e}'})
    audit={'index_records':len(index),'valid_records':len(records),'excluded':exclusions,'duplicate_index_entries':duplicates,
           'runs':len({r['run'] for r in records}),'explicit_purpose_or_expectation':sum(any(k in r['original_intention'] for k in ['action_purpose','expected_effect','expected_outcome','expected_stop_state']) for r in records),
           'filter':'Physical flag + grasp command + readable hash-matching Before/After; no filter on DONE, area, intended success or prior discoveries.',
           'raw_input_scope':'Index metadata, raw dispatch_intent/execution/native_plan, raw Before/After only. No previous outcome analyses or human skill reports.'}
    save(root/'experience_index.json',records);save(root/'data_audit.json',audit)
    return records,audit
