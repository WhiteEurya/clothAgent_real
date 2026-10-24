"""Expand frozen DINO search to every original demonstration RGB.

All-image ranking is independent of action availability. A second ranking contains
only confirmed pre-images with recorded Pa/Pb. No GPT or robot calls occur here.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw

from cloth_agent.dino_correspondence import DEFAULT_MODEL, save_json, sha, valid_patches
from .dino_reference_baseline import JointEncoder, rank_nodes


def inventory(root):
    root=Path(root).resolve()
    paths=sorted(Path(p) for p in subprocess.check_output(
        ['rg','--files','--hidden','--no-ignore',str(root),'-g','rgb.png'],text=True).splitlines())
    records={}
    for path in paths:
        digest=sha(path)
        role=('action_before' if path.parent.name=='observation' and path.parent.parent.name.startswith('traj_')
              else 'action_after' if path.parent.name=='after' and path.parent.parent.name.startswith('traj_')
              else 'reset' if 'resets' in path.relative_to(root).parts else 'pending_or_other')
        with Image.open(path) as im:
            size=list(im.size);im.verify()
        entry=records.setdefault(digest,dict(sha256=digest,original_wh=size,aliases=[]))
        entry['aliases'].append(dict(path=str(path),role=role))
    return list(records.values()),dict(root=str(root),file_count=len(paths),
        unique_file_hashes=len(records),roles=dict(Counter(a['role'] for r in records.values() for a in r['aliases'])))


def point_pair(action):
    points={p['name']:p.get('pixel_xy') for p in action.get('selected_points',[])}
    if set(points)!={'start','target'}:return False
    a,b=np.asarray(points['start'],float),np.asarray(points['target'],float)
    return a.shape==(2,) and b.shape==(2,) and np.isfinite([a,b]).all() and np.linalg.norm(a-b)>=1


def make_action_refs(archive, entries, old_refs):
    by_hash={r['source_image']['sha256']:r for r in old_refs}
    refs=[];excluded=[]
    for step in archive['steps']:
        action=step['commanded_action']
        if not point_pair(action):
            excluded.append(dict(step_id=step['step_id'],folder=step['source_folder'],reason='no_unique_Pa_Pb_pair'));continue
        rgb=archive['observations'][step['before_observation']]['rgb']
        entry=entries[rgb['sha256']]
        points=[p['pixel_xy'] for p in action['selected_points']]
        if any(np.any(np.asarray(p)<0) or np.any(np.asarray(p)>=entry['geometry']['original_wh']) for p in points):
            excluded.append(dict(step_id=step['step_id'],folder=step['source_folder'],reason='point_outside_original_image'));continue
        previous=by_hash.get(rgb['sha256'])
        sample_id=f"{step['session']}/{Path(step['source_folder']).name}"
        edge=copy.deepcopy(previous['edge']) if previous else dict(
            edge_id='raw_'+sample_id.replace('/','_'),source_node='raw_'+sample_id.replace('/','_'),
            target_node=None,action_family='recorded Pa -> Pb',seed_step_id=step['step_id'],
            seed_action=dict(exact_command=action,source_observation=step['before_observation']),
            replayable=False,replay_block_reason='Raw recorded point pair; visual reference only, requires current grounding and execution preflight.')
        edge['seed_action']=dict(exact_command=action,source_observation=step['before_observation'])
        edge['seed_step_id']=step['step_id']
        refs.append(dict(node_id='image_'+rgb['sha256'][:16],edge_id=edge['edge_id'],edge=edge,
            graph_node_id=previous['node_id'] if previous else None,
            step_id=step['step_id'],observation_id=step['before_observation'],sample_id=sample_id,
            action_family=edge['action_family'],source_image=rgb,
            image='../cache/'+entry['image'],features='../cache/'+entry['features'],
            features_sha256=entry['features_sha256'],geometry=entry['geometry']))
    return refs,excluded


def sheet(current, ranked, cache, output):
    rows=[dict(path=str(current),label='CURRENT',detail='Saved round 002 observation')]
    for c in ranked[:9]:
        detail=c.get('display_label',c.get('sample_id',c['node_id']))
        if len(detail)>75:detail=detail[:36]+'...'+detail[-36:]
        paired=c.get('has_confirmed_point_pair',True)
        rows.append(dict(path=str(cache/c['image']),
            label=f"#{c['rank']} cosine={c['score']:.4f} | Pa/Pb: {'available' if paired else 'unavailable'}",detail=detail))
    width,height=480,300
    canvas=Image.new('RGB',(width*2,height*((len(rows)+1)//2)), 'white');draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        x,y=(i%2)*width,(i//2)*height
        with Image.open(row['path']) as im:
            im=im.convert('RGB');im.thumbnail((width,height-36));canvas.paste(im,(x,y+36))
        draw.text((x+7,y+5),row['label'],fill='black')
        draw.text((x+7,y+19),row['detail'],fill='black')
    canvas.save(output)


def run(data_root, previous_cache, previous_query, output):
    from scripts.compile_arm6_archive_graph import import_archive
    started=time.perf_counter()
    data_root,previous_cache,previous_query,output=map(lambda p:Path(p).resolve(),
        (data_root,previous_cache,previous_query,output))
    old=json.loads((previous_cache/'manifest.json').read_text())
    original_query=json.loads((previous_query/'evidence.json').read_text())
    if sha(previous_cache/'manifest.json')!=original_query['manifest_sha256']:
        raise ValueError('Original query/cache mismatch')
    if sha(previous_query/'I_current.png')!=original_query['current']['sha256'] or sha(previous_query/'current_features.npz')!=original_query['current']['features_sha256']:
        raise ValueError('Current RGB or cached encoding changed')
    images,summary=inventory(data_root)
    archive=import_archive(data_root)
    output.mkdir(parents=True,exist_ok=False)
    cache=output/'cache';(cache/'features').mkdir(parents=True);(cache/'images').mkdir()
    save_json(output/'archive_snapshot.json',archive)
    encoder=JointEncoder(DEFAULT_MODEL,device='cpu')
    if encoder.contract!=old['contract'] or encoder.contract!=original_query['contract']:
        raise ValueError('Expanded/current/original encoder contracts differ')
    old_by_hash={r['source_image']['sha256']:r for r in old['references']}
    entries={};cls_by_hash={};reused=0
    for i,item in enumerate(images):
        digest=item['sha256'];source=Path(item['aliases'][0]['path'])
        rgb=cache/'images'/f'{digest}.png';feature=cache/'features'/f'{digest}.npz'
        shutil.copyfile(source,rgb)
        if sha(rgb)!=digest:raise ValueError('Source changed during snapshot')
        if digest in old_by_hash:
            old_ref=old_by_hash[digest];old_feature=previous_cache/old_ref['features']
            if sha(old_feature)!=old_ref['features_sha256']:raise ValueError('Old feature hash changed')
            shutil.copyfile(old_feature,feature)
            with np.load(feature,allow_pickle=False) as values:cls=values['cls'].copy()
            geometry=old_ref['geometry'];reused+=1
        else:
            with Image.open(rgb) as im:cls,patches,geometry,_=encoder.encode_joint(im.convert('RGB'))
            np.savez_compressed(feature,cls=cls,patches=patches,valid=valid_patches(geometry))
        entry=dict(**item,node_id='image_'+digest[:16],edge_id='none',
            image=str(rgb.relative_to(cache)),features=str(feature.relative_to(cache)),
            features_sha256=sha(feature),source_image=dict(path=str(source),sha256=digest),geometry=geometry)
        entry['display_label']=item['aliases'][0]['role']+' '+source.relative_to(data_root).as_posix()
        entries[digest]=entry;cls_by_hash[digest]=cls
        if (i+1)%40==0 or i+1==len(images):print(f"Corpus {i+1}/{len(images)}; reused {reused}; new encodings {encoder.calls}",flush=True)
    all_refs=list(entries.values());all_cls=np.stack([cls_by_hash[r['sha256']] for r in all_refs])
    np.save(cache/'cls.npy',all_cls)
    action_refs,excluded=make_action_refs(archive,entries,old['references'])
    if not action_refs:raise ValueError('No eligible Pa/Pb reference')
    action_hashes={r['source_image']['sha256'] for r in action_refs}
    for r in all_refs:r['has_confirmed_point_pair']=r['sha256'] in action_hashes
    save_json(cache/'manifest.json',dict(contract=encoder.contract,references=all_refs,inventory=summary,
        reused_reference_encodings=reused,new_reference_encodings=encoder.calls,cls_sha256=sha(cache/'cls.npy')))
    action_cache=output/'action_cache';action_cache.mkdir()
    action_cls=np.stack([cls_by_hash[r['source_image']['sha256']] for r in action_refs]);np.save(action_cache/'cls.npy',action_cls)
    manifest=dict(version=2,contract=encoder.contract,references=action_refs,cls_sha256=sha(action_cache/'cls.npy'),
        scope='All confirmed original Arm6 pre-images with recorded Pa/Pb; no graph membership or goal filter',
        ranking='exact global CLS cosine; unique RGB hashes; ties image id then edge id',
        unique_images=len(action_hashes),node_count=len(action_hashes),archive_snapshot_sha256=sha(output/'archive_snapshot.json'),
        source_root=str(data_root),excluded_point_pairs=excluded,inventory=summary)
    save_json(action_cache/'manifest.json',manifest)
    with np.load(previous_query/'current_features.npz',allow_pickle=False) as values:current_cls=values['cls'].copy()
    search_started=time.perf_counter();all_ranked=rank_nodes(all_refs,all_cls,current_cls);all_search_s=time.perf_counter()-search_started
    search_started=time.perf_counter();actions_ranked=rank_nodes(action_refs,action_cls,current_cls);action_search_s=time.perf_counter()-search_started
    save_json(output/'all_images_ranking.json',all_ranked);save_json(output/'action_images_ranking.json',actions_ranked)
    current=previous_query/'I_current.png'
    sheet(current,all_ranked,cache,output/'all_images_top9.jpg')
    sheet(current,actions_ranked,action_cache,output/'action_images_top9.jpg')
    # Compatible evidence for a subsequent fixed-reference grounding call.
    query=output/'retrieval';query.mkdir()
    shutil.copyfile(current,query/'I_current.png');shutil.copyfile(previous_query/'current_features.npz',query/'current_features.npz')
    evidence=copy.deepcopy(original_query)
    evidence.update(cache=str(action_cache),manifest_sha256=sha(action_cache/'manifest.json'),
        top3=actions_ranked[:3],selected=actions_ranked[0],ranking=manifest['ranking'],scope=manifest['scope'],
        current_encoded_images=0,current_forward_calls=0,online_reference_forward_calls=0,
        current_features_reused_from=str(previous_query/'evidence.json'),
        score_gap_top1_top2=actions_ranked[0]['score']-actions_ranked[1]['score'],
        timing=dict(model_load_s=0.,decode_preprocess_forward_s=0.,forward_s=0.,reference_cls_load_s=0.,
            exact_ranking_s=action_search_s,time_to_selection_s=action_search_s,query_and_artifacts_s=None,
            definition='Cached current features reused; no current encoding or model load. Only exact ranking timed; corpus expansion is offline.'))
    evidence['current']['source']=str(current)
    for c in evidence['top3']:shutil.copyfile(action_cache/c['image'],query/f"reference_{c['rank']}.png")
    save_json(query/'ranking.json',actions_ranked);save_json(query/'selected_action_edge.json',actions_ranked[0]['edge'])
    # A numeric duration remains available for the existing visualization reader.
    evidence['timing']['query_and_artifacts_s']=time.perf_counter()-search_started
    save_json(query/'evidence.json',evidence)
    old_digest=original_query['selected']['source_image']['sha256']
    result=dict(**summary,confirmed_archive_steps=len(archive['steps']),eligible_point_pair_images=len(action_hashes),
        reused_reference_encodings=reused,new_reference_encodings=encoder.calls,current_new_encodings=0,
        original_winner_rank_all=next(r['rank'] for r in all_ranked if r['sha256']==old_digest),
        original_winner_rank_actions=next((r['rank'] for r in actions_ranked if r['source_image']['sha256']==old_digest),None),
        best_all=dict(rank=1,score=all_ranked[0]['score'],source_image=all_ranked[0]['source_image'],
                      aliases=all_ranked[0]['aliases'],has_confirmed_point_pair=all_ranked[0]['has_confirmed_point_pair']),
        best_action=dict(score=actions_ranked[0]['score'],sample_id=actions_ranked[0]['sample_id'],edge_id=actions_ranked[0]['edge_id']),
        original_top1_score=original_query['selected']['score'],all_search_s=all_search_s,action_search_s=action_search_s,
        elapsed_including_offline_expansion_s=time.perf_counter()-started,robot_calls=0,gpt_calls=0)
    save_json(output/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/demonstrations'))
    parser.add_argument('--previous-cache',type=Path,required=True)
    parser.add_argument('--previous-query',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
