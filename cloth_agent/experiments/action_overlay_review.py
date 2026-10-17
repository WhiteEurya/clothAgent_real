"""Offline action-only review of frozen Stage 1, with calibrated Pa/Pb overlays."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .discovery_data import read, save, digest, file_hash, stage_sample
from .discovery_goal import model_task_goal
from .decision_state_discovery import Requests, batches, montage


def frozen(path, canonical_hash=False):
    path=Path(path);v=read(path)
    actual=digest(v) if canonical_hash else file_hash(path)
    if actual!=read(path.with_suffix('.sha256.json')).get('sha256'):
        raise ValueError(f'frozen artifact hash mismatch: {path}')
    return v


def project(points, intrinsics, transform):
    points=np.asarray(points,dtype=float);K=np.asarray(intrinsics,dtype=float);X=np.asarray(transform,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or K.shape!=(3,3) or X.shape!=(4,4):raise ValueError('invalid projection dimensions')
    if not all(np.isfinite(a).all() for a in (points,K,X)):raise ValueError('nonfinite projection inputs')
    if not np.allclose(X[:3,:3].T@X[:3,:3],np.eye(3),atol=1e-5):raise ValueError('nonrigid camera rotation')
    # Historical X already includes manual base offsets. Never add them twice.
    camera=(points/1000-X[:3,3])@X[:3,:3]
    if (camera[:,2]<=1e-6).any():raise ValueError('point behind camera')
    q=camera@K.T
    return q[:,:2]/q[:,2,None]


def projection(sample):
    before=Path(sample['before_rgb']);root=before.parent
    if file_hash(before)!=sample['image_hashes']['before']:raise ValueError('before image hash changed')
    mp=root/'camera_A_upright_mapping.json';mapping=read(mp)
    if mapping.get('rotation')!='clockwise90':raise ValueError('missing supported camera mapping')
    guide_path=Path(mapping['coordinate_guide']);guide=read(guide_path)
    if guide.get('coordinate_frame')!='robot_base_mm':raise ValueError('unknown guide coordinate frame')
    result_path=guide_path.parent/'result.json'
    views=[v for v in read(result_path).get('views',[]) if v.get('label')=='A']
    if len(views)!=1:raise ValueError('missing unique Camera A calibration')
    view=views[0];raw_path=guide_path.parent/view['image']
    with Image.open(raw_path) as im:raw=im.convert('RGB')
    with Image.open(before) as im:display=im.convert('RGB')
    if list(raw.size)!=mapping['raw_size_xy'] or list(display.size)!=mapping['upright_size_xy']:
        raise ValueError('camera mapping image size mismatch')
    if not np.array_equal(np.asarray(raw.transpose(Image.Transpose.ROTATE_270)),np.asarray(display)):
        raise ValueError('calibration raw RGB does not match before image')
    refs=guide.get('samples',[])
    if len(refs)<3:raise ValueError('not enough coordinate references to verify calibration')
    uv=project([r['base_xyz_mm'] for r in refs],view['intrinsics'],view['X_base_camera'])
    errors=np.linalg.norm(uv-np.asarray([r['pixel_xy'] for r in refs]),axis=1)
    if errors.max()>1:raise ValueError('coordinate-guide reprojection exceeds 1 pixel')
    action=sample['action']
    if action['grasp'] is None or action['target'] is None:raise ValueError('ambiguous Pa/Pb')
    uv=project([action['grasp'],action['target']],view['intrinsics'],view['X_base_camera'])
    display_uv=np.column_stack((raw.height-1-uv[:,1],uv[:,0]))
    inside=((display_uv>=0)&(display_uv<np.array(display.size))).all(axis=1)
    if not inside.all():raise ValueError('projected endpoint outside before image; no clamping')
    return dict(status='AVAILABLE',Pa_pixel_xy=display_uv[0].tolist(),Pb_pixel_xy=display_uv[1].tolist(),
        displacement_pixel_xy=(display_uv[1]-display_uv[0]).tolist(),
        displacement_mm=action['normalized']['delta'],image_size_xy=list(display.size),
        endpoint_semantics='Commanded Cartesian endpoints, not measured contact; arrow is not full path.',
        calibration_check=dict(reference_count=len(refs),max_reprojection_error_px=float(errors.max()),
                               rotated_rgb_exact_match=True),
        calibration=dict(intrinsics=view['intrinsics'],X_base_camera=view['X_base_camera'],rotation='clockwise90'),
        sources=[dict(file=str(f),sha256=file_hash(f)) for f in (mp,guide_path,result_path,raw_path)])


def draw_overlay(before, target, sid, p):
    with Image.open(before) as im:im=im.convert('RGB')
    d=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('DejaVuSans.ttf',22)
    except OSError:font=ImageFont.load_default()
    d.rectangle((0,0,im.width,65),fill='black')
    d.text((8,4),sid,fill='white',font=font)
    if p['status']!='AVAILABLE':
        d.text((8,32),'Projection unavailable; see numeric action',fill='white',font=font)
    else:
        d.text((8,32),'Pa cyan -> Pb magenta | commanded endpoints',fill='white',font=font)
        a=np.array(p['Pa_pixel_xy']);b=np.array(p['Pb_pixel_xy']);v=b-a;n=np.linalg.norm(v)
        d.line([tuple(a),tuple(b)],fill='yellow',width=4)
        if n>1:
            u=v/n;normal=np.array([-u[1],u[0]])
            d.polygon([tuple(b),tuple(b-18*u+8*normal),tuple(b-18*u-8*normal)],fill='yellow')
        for point,label,color in ((a,'Pa','cyan'),(b,'Pb','magenta')):
            x,y=point;d.ellipse((x-9,y-9,x+9,y+9),outline=color,width=4)
            d.text((min(x+12,im.width-45),max(68,min(y-25,im.height-28))),label,fill=color,stroke_width=2,stroke_fill='black',font=font)
    Path(target).parent.mkdir(parents=True,exist_ok=True);im.save(target)


def prepare(source, output):
    source=Path(source);output=Path(output)
    m=frozen(source/'manifest.json',True);tax=frozen(source/'stage1/taxonomy_v0.json')
    assignments=frozen(source/'stage1/assignments.json');config=read(source/'config.json')
    if 'target_geometry' not in {f['id'] for f in tax['factors']}:raise ValueError('expected frozen target_geometry factor')
    ids=sorted(m['selected_sample_ids']);random.Random(config['seed']).shuffle(ids)
    all_samples={s['sample_id']:s for s in m['samples']};by_id={a['sample_id']:a for a in assignments}
    rows=[];excluded=[]
    for sid in ids:
        s=all_samples[sid]
        if not s['eligible']['action']:
            excluded.append(dict(sample_id=sid,reason='No supported historical action'));continue
        # Explicit projection inputs only. Do not read after_rgb, outcome, evaluation or execution artifacts.
        inputs={k:s[k] for k in ('before_rgb','image_hashes','action')}
        try:p=projection(inputs)
        except (ValueError,OSError,KeyError,TypeError) as exc:p=dict(status='UNAVAILABLE',reason=str(exc))
        row=dict(sample_id=sid,run_id=s['run_id'],iteration_id=s['iteration_id'],
                 before_rgb=s['before_rgb'],before_sha256=s['image_hashes']['before'],
                 action=stage_sample(s,'action')['action'],projection=p,image_only_assignment=by_id[sid])
        rows.append(row)
        draw_overlay(s['before_rgb'],output/'overlays'/f'{sid}.png',sid,p)
    plan=dict(protocol='frozen-action-overlay-v1',source=str(source.resolve()),source_manifest_hash=digest(m),
        frozen_taxonomy=tax,frozen_taxonomy_hash=digest(tax),task_goal=model_task_goal(m),
        model=config['model'],seed=config['seed'],batch_size=config['batch_size'],samples=rows,excluded=excluded,
        limitations=['Historical commands are not measured contacts or verified good actions.',
            'Calibration check verifies recorded coordinate consistency, not physical calibration accuracy.',
            'Source manifest is previously extracted; outcome fields are never selected, copied or passed to the model.',
            'No garment-relative registration is asserted; cross-run XYZ comparisons require caution.'])
    save(output/'manifest.json',plan,immutable=True)
    save(output/'frozen_taxonomy.json',tax,immutable=True)
    save(output/'frozen_taxonomy.sha256.json',dict(sha256=file_hash(output/'frozen_taxonomy.json')),immutable=True)
    return plan


def model_row(row):
    p=row['projection']
    return dict(sample_id=row['sample_id'],action=row['action'],image_only_assignment=row['image_only_assignment'],
        projection={k:p[k] for k in ('status','reason','Pa_pixel_xy','Pb_pixel_xy','displacement_pixel_xy',
                    'displacement_mm','image_size_xy','endpoint_semantics') if k in p})


def run(plan, output, requests):
    output=Path(output);rows=plan['samples'];lookup={r['sample_id']:r for r in rows};size=plan['batch_size']
    def payload(subset):
        return dict(sample_ids=[r['sample_id'] for r in subset],samples=[model_row(r) for r in subset],
            task_goal=plan['task_goal'],frozen_taxonomy=plan['frozen_taxonomy'],
            frozen_taxonomy_hash=plan['frozen_taxonomy_hash'])
    def ask(kind, subset, extra=None, sheets=False):
        p=payload(subset);p.update(extra or {});images=[];image_ids=[]
        if sheets:
            for i,chunk in enumerate(batches(subset,size)):
                ids=[r['sample_id'] for r in chunk];path=output/'montages'/f'all_actions_{i}.png'
                montage(ids,{sid:output/'overlays'/f'{sid}.png' for sid in ids},path)
                images.append((f'actions_{i}',path));image_ids.append(dict(image_id=f'actions_{i}',sample_ids=ids,role='before_action_overlay_sheet'))
        else:
            for r in subset:
                sid=r['sample_id']
                for ident,path,role in [(sid,Path(r['before_rgb']),'before'),(sid+'_action',output/'overlays'/f'{sid}.png','before_action_overlay')]:
                    images.append((ident,path));image_ids.append(dict(image_id=ident,sample_id=sid,role=role))
        p['image_ids']=image_ids
        return requests.ask('action',kind,p,images,plan['frozen_taxonomy_hash'])
    nodes=[]
    # Original + overlay per sample: <= batch_size image attachments, default 12.
    for i,chunk in enumerate(batches(rows,max(1,size//2))):
        v=ask('frozen_review',chunk);save(output/'stage2'/f'batch_{i:03d}.json',v,immutable=True);nodes.append(v)
    v=ask('frozen_aggregate',rows,dict(child_summaries=nodes),sheets=True)
    save(output/'stage2/aggregate_initial.json',v,immutable=True)
    pairs=v['appearance_similar_action_different']+v['appearance_different_action_equivalent']
    checks=[]
    # Verify all proposed pairs at full resolution; pair members never split across requests.
    for i,chunk in enumerate(batches(pairs,max(1,size//4))):
        ids=list(dict.fromkeys(s for pair in chunk for s in pair['samples']))
        check=ask('frozen_pairs',[lookup[s] for s in ids],dict(proposed_pairs=chunk))
        save(output/'stage2'/f'pair_check_{i:03d}.json',check,immutable=True);checks.append(check)
    if checks:
        v=ask('frozen_aggregate',rows,dict(child_summaries=nodes,initial_review=v,full_resolution_pair_checks=checks),sheets=True)
    save(output/'stage2/action_review.json',v,immutable=True)
    frozen(output/'frozen_taxonomy.json')
    return v


def report(plan, output, requests=None, error=None):
    output=Path(output);v=read(output/'stage2/action_review.json')
    calls=[__import__('json').loads(x) for x in (output/'requests/calls.jsonl').read_text().splitlines()] if (output/'requests/calls.jsonl').exists() else []
    checks=[read(f) for f in (output/'stage2').glob('pair_check_*.json')]
    pair_audit={}
    for field in ('appearance_similar_action_different','appearance_different_action_equivalent'):
        checked={tuple(sorted(pair['samples'])) for c in checks for pair in c[field]}
        pair_audit[field]=[dict(samples=pair['samples'],
            present_in_full_resolution_pair_review=tuple(sorted(pair['samples'])) in checked,
            note='Explicit pair-review evidence exists' if tuple(sorted(pair['samples'])) in checked
                 else 'Final aggregation claim; no explicit pair-review entry. Consult batch evidence.')
            for pair in v.get(field,[])]
    r=dict(status='COMPLETED' if v else 'FAILED' if error else 'PREPARED',error=error,
        sample_count=len(plan['samples']),projection_counts={s:sum(r['projection']['status']==s for r in plan['samples']) for s in ('AVAILABLE','UNAVAILABLE')},
        source_manifest_hash=plan['source_manifest_hash'],frozen_taxonomy_hash=plan['frozen_taxonomy_hash'],
        model_calls=sum(not x['cache_hit'] for x in calls),cache_hits=sum(x['cache_hit'] for x in calls),
        elapsed_model_s=sum(x['elapsed_s'] for x in calls),review=v,
        counterexample_verification=pair_audit,limitations=plan['limitations'])
    save(output/'report.json',r)
    lines=['# Frozen Stage 1 action-overlay review','',f'Status: {r["status"]}; samples: {r["sample_count"]}; model calls: {r["model_calls"]}.',
        '', 'Stage 1 taxonomy is unchanged. No outcomes or current-sample after observations were supplied.',
        'Cyan: commanded Pa; magenta: commanded Pb; yellow: endpoint displacement, not the full trajectory.',
        '', '## Counterexample verification (host audit)', '',
        'Final aggregation can add pairs. Presence in an explicit full-resolution pair-check is recorded separately; it is not physical ground truth.',
        '```json',__import__('json').dumps(pair_audit,ensure_ascii=False,indent=2),'```',
        '', '## Runtime GPT evidence', '', '```json',__import__('json').dumps(v,ensure_ascii=False,indent=2),'```','', '## Overlays','']
    for row in plan['samples']:lines += [f'### {row["sample_id"]}',f'![Pa/Pb](overlays/{row["sample_id"]}.png)','']
    lines+=['## Limitations','']+['- '+x for x in plan['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return r


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--max-calls',type=int,default=16);ap.add_argument('--no-cache',action='store_true')
    a=ap.parse_args(argv)
    if a.source.resolve()==a.output.resolve():ap.error('use a separate action-review output directory')
    if a.max_calls<1:ap.error('max-calls must be positive')
    plan=prepare(a.source,a.output);r=report(plan,a.output)
    print({k:r[k] for k in ('status','sample_count','projection_counts','source_manifest_hash')},flush=True)
    if not a.execute:return 0
    from .discovery_model import CodexMultimodal
    req=Requests(a.output,plan['model'],CodexMultimodal(plan['model']),a.no_cache,max_calls=a.max_calls)
    error=None
    try:run(plan,a.output,req)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';raise
    finally:
        report(plan,a.output,req,error)
        frozen(a.source/'stage1/taxonomy_v0.json');frozen(a.source/'stage1/assignments.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
