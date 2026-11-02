"""Resume cached requests after one scan transport failure, without retrying it.

The original runner's pool.map raises on a failed scan and waits for its already
queued work. This adapter is used only AFTER that process exits. No old result is
overwritten; unsuccessful scan batches contribute no invented nominations.
Selection rules, prompts, images, splits and skill budget are unchanged.
"""
import time
from pathlib import Path
import run as original

invoke_original=original.invoke
finish_original=original.finish

def invoke(name,prompt,payload,images,schema):
    p=original.ROOT/'calls'/name/'result.json'
    if name.startswith('scan_') and p.exists():
        failed=original.read(p)
        if failed.get('status')!='COMPLETE':
            return {'candidates':[], 'screening_limitations':'API_TRANSPORT_FAILED; these pairs were NOT visually screened. No retry, no semantic rejection. '+failed.get('error','')}
    try:
        return invoke_original(name,prompt,payload,images,schema)
    except RuntimeError:
        if name.startswith('scan_') and p.exists() and original.read(p).get('status')!='COMPLETE':
            return {'candidates':[], 'screening_limitations':'API_TRANSPORT_FAILED; no retry and no semantic rejection. '+original.read(p).get('error','')}
        raise

def finish(summary,rs):
    sheets=original.read(original.ROOT/'contact_sheets.json')
    missing=[]
    for i in range(0,len(sheets),3):
        p=original.ROOT/'calls'/f'scan_{i//3:02d}'/'result.json'
        if not p.exists() or original.read(p).get('status')!='COMPLETE':
            missing += [id for page in sheets[i:i+3] for id in page['ids']]
    summary.update(unscreened_due_to_transport=len(missing),visually_screened_pairs=summary['discovery_pairs']-len(missing),transport_recovery='Failed contact scan retained without retry; its candidates are empty, not negative labels. Ranking/prompt/split unchanged.')
    original.save(original.ROOT/'unscreened_pairs.json',missing)
    finish_original(summary,rs)

if __name__=='__main__':
    original.save(original.ROOT/'transport_recovery.json',{'time':time.time(),'adapter':str(Path(__file__).resolve()),'adapter_sha256':original.sha(__file__),'reason':'A failed scan raised from pool.map; resume cached successes, skip failed scans without retry or invented labels.','changes_to_selection_or_prompts':False,'new_induction_round':False})
    original.invoke=invoke
    original.finish=finish
    original.main()
