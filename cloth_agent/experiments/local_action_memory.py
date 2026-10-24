"""Offline local action memory: grasp crops, exact demo outcomes, DINO/HNSW.

No robot connection. Graph nodes are provenance only, never retrieval labels.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .discovery_data import save, file_hash
from .graph_policy import checked_file, evidence_file

PROTOCOL = 'arm6-local-action-memory-v1'


def crop_box(size, point, fraction):
    w, h = size
    x, y = point
    if not (0 < fraction <= 1 and 0 <= x < w and 0 <= y < h):
        raise ValueError('invalid crop size or anchor')
    side = max(16, min(min(size), round(min(size)*fraction)))
    left = max(0, min(w-side, round(x-side/2)))
    top = max(0, min(h-side, round(y-side/2)))
    return [left, top, left+side, top+side]


def query_boxes(size, fractions, stride=.5):
    if not 0 < stride <= 1: raise ValueError('invalid stride')
    w, h = size
    result = []
    for fraction in fractions:
        side = crop_box(size, (0,0), fraction)[2]
        step = max(1, round(side*stride))
        xs = sorted(set([*range(0,w-side+1,step), w-side]))
        ys = sorted(set([*range(0,h-side+1,step), h-side]))
        result.extend([x,y,x+side,y+side] for y in ys for x in xs)
    return list(dict.fromkeys(tuple(b) for b in result))


def export(graph_path, output, session_prefix='20261019', fractions=(.22,.44)):
    from .arm6_traj_replay import compile_templates
    graph, trajectory, templates = compile_templates(Path(graph_path), allow_partial=True)
    steps = {s['step_id']:s for s in trajectory['steps']}
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    actions = {}; crops = []; excluded = {}
    for edge in graph['edges']:
        eid = edge['edge_id']; step = steps[edge['seed_step_id']]
        if not step['session'].startswith(session_prefix):
            excluded[eid] = 'outside explicitly selected unfolding collection cohort'; continue
        if eid not in templates:
            excluded[eid] = graph['replay_unavailable_edges'][eid]; continue
        before_id = step['before_observation']; after_id = step['after_observation']
        if before_id != edge['seed_action']['source_observation'] or not after_id:
            excluded[eid] = 'missing exact paired before/after'; continue
        before = trajectory['observations'][before_id]['rgb']
        after = trajectory['observations'][after_id]['rgb']
        checked_file(after)
        template = templates[eid]
        with Image.open(checked_file(before)) as im:
            im = im.convert('RGB'); size = im.size
            for fraction in fractions:
                box = crop_box(size,template['reference_start_px'],fraction)
                crop = im.crop(box)
                for angle in (0,90,180,270):
                    cid = f'{eid}_{round(fraction*100):02d}_{angle:03d}'
                    path = output/'crops'/f'{cid}.png'; path.parent.mkdir(exist_ok=True)
                    crop.rotate(angle,expand=True).save(path)
                    crops.append(dict(crop_id=cid,edge_id=eid,box_xyxy=box,
                                      rotation_ccw_deg=angle,**evidence_file(path)))
        actions[eid] = dict(edge_id=eid,session=step['session'],seed_step_id=step['step_id'],
            before=before,after=after,action_family=edge['action_family'],
            action_template=edge['action_template'],observed_outcome=edge['observed_outcome'],
            source_evidence_limitations=edge['action_template'].get('other_constraints',[]),
            seed_outcome=edge['seed_outcome'],template=template,
            reference_start_px=template['reference_start_px'],reference_target_px=template['reference_target_px'])
    if not actions: raise ValueError('empty local action library')
    manifest = dict(protocol=PROTOCOL,graph=evidence_file(graph_path),goal='unfold',
        selection=dict(session_prefix=session_prefix,
            basis='User-selected unfolding collection cohort; individual action success is NOT assumed. Inward moves retained.'),
        fractions=list(fractions),actions=actions,crops=crops,excluded=excluded,
        pairing='Exact trajectory before/after; never merged-node representative substitution',
        robot_calls=0,graph_state_matching=False)
    save(output/'library.json',manifest)
    return dict(actions=len(actions),crops=len(crops),excluded=len(excluded),library=str(output/'library.json'))


def load_library(directory):
    directory=Path(directory)
    lib=json.loads((directory/'library.json').read_text())
    if lib['protocol']!=PROTOCOL or lib['goal']!='unfold':raise ValueError('wrong local action library')
    checked_file(lib['graph'])
    for a in lib['actions'].values():
        checked_file(a['before']);checked_file(a['after'])
    return lib


def encode_batches(encoder, images, batch_size=16):
    vectors=[];preprocess=forward=0.
    for start in range(0,len(images),batch_size):
        v,p,f=encoder.encode(images[start:start+batch_size])
        vectors.append(v);preprocess+=p;forward+=f
    return np.concatenate(vectors),dict(dino_preprocess_s=preprocess,dino_forward_s=forward)


def embed(directory, model_path, device='cuda'):
    import faiss
    from .visual_retrieval_index import DinoEncoder,build_index
    directory=Path(directory);lib=load_library(directory)
    if (directory/'index_manifest.json').exists():raise FileExistsError('index already exists; use a new library directory')
    encoder=DinoEncoder(model_path,device);images=[]
    for row in lib['crops']:
        with Image.open(checked_file(row)) as im:images.append(im.convert('RGB'))
    vectors,timing=encode_batches(encoder,images)
    np.save(directory/'local_embeddings.npy',vectors,allow_pickle=False)
    faiss.write_index(build_index(vectors),str(directory/'local.index'))
    save(directory/'index_manifest.json',dict(library_sha256=file_hash(directory/'library.json'),
        encoder=encoder.meta,dimension=vectors.shape[1],references=len(vectors),timing=timing,
        artifacts={n:file_hash(directory/n) for n in ['local_embeddings.npy','local.index']}))
    return dict(status='PRECOMPUTED',actions=len(lib['actions']),reference_crops=len(vectors),timing=timing)


def distinct_actions(scores, ids, boxes, crops, top_k, excluded=()):
    best={}
    for q,(ss,ii) in enumerate(zip(scores,ids)):
        for score,index in zip(ss,ii):
            if index<0:continue
            ref=crops[int(index)];eid=ref['edge_id']
            if eid in excluded or not math.isfinite(float(score)):continue
            if eid not in best or score>best[eid]['similarity']:
                best[eid]=dict(edge_id=eid,similarity=float(score),query_box_xyxy=list(boxes[q]),
                               reference_crop=ref,query_window_index=q)
    return sorted(best.values(),key=lambda r:(-r['similarity'],r['edge_id']))[:top_k]


def search(directory, image, output, top_k=6, exclude_edges=()):
    import faiss
    from .visual_retrieval_index import DinoEncoder,encoder_contract
    if not 1<=top_k<=12:raise ValueError('top_k must be 1..12')
    started=time.perf_counter();directory=Path(directory);lib=load_library(directory)
    meta=json.loads((directory/'index_manifest.json').read_text())
    if file_hash(directory/'library.json')!=meta['library_sha256']:raise ValueError('library changed since indexing')
    for n,digest in meta['artifacts'].items():
        if file_hash(directory/n)!=digest:raise ValueError('local index artifact changed')
    faiss.omp_set_num_threads(1);index=faiss.read_index(str(directory/'local.index'))
    if index.ntotal!=len(lib['crops']) or index.d!=meta['dimension']:raise ValueError('index dimensions mismatch')
    encoder=DinoEncoder(meta['encoder']['path'],meta['encoder']['device'])
    if encoder_contract(encoder.meta)!=encoder_contract(meta['encoder']):raise ValueError('encoder mismatch')
    setup=time.perf_counter()-started;t=time.perf_counter()
    with Image.open(image) as im:im=im.convert('RGB')
    boxes=query_boxes(im.size,lib['fractions']);patches=[im.crop(b) for b in boxes]
    decode=time.perf_counter()-t
    vectors,timing=encode_batches(encoder,patches)
    t=time.perf_counter();scores,ids=index.search(vectors,min(index.ntotal,max(32,top_k*8)))
    rank=distinct_actions(scores,ids,boxes,lib['crops'],top_k,set(exclude_edges))
    timing.update(setup_s=setup,decode_and_crop_s=decode,ann_search_and_aggregation_s=time.perf_counter()-t,
                  total_s=time.perf_counter()-started)
    result=dict(query=evidence_file(image),ranking=rank,query_windows=len(boxes),timing=timing,
        reference_encode_calls_runtime=0,graph_state_matching=False,
        scope='Multi-scale sliding RGB crops; rotated grasp crops in offline memory; uncalibrated similarity, not probability.')
    save(output,result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    x=subs.add_parser('export');x.add_argument('--graph',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
    x.add_argument('--session-prefix',default='20261019')
    x=subs.add_parser('embed');x.add_argument('--library',type=Path,required=True)
    x.add_argument('--model-path',type=Path,default=Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small'))
    x.add_argument('--device',default='cuda')
    x=subs.add_parser('search');x.add_argument('--library',type=Path,required=True);x.add_argument('--image',type=Path,required=True)
    x.add_argument('--output',type=Path,required=True);x.add_argument('--top-k',type=int,default=6)
    x.add_argument('--exclude-edges',default='')
    a=p.parse_args(argv)
    if a.command=='export':r=export(a.graph,a.output,a.session_prefix)
    elif a.command=='embed':r=embed(a.library,a.model_path,a.device)
    else:r=search(a.library,a.image,a.output,a.top_k,[e for e in a.exclude_edges.split(',') if e])
    print(json.dumps(r,ensure_ascii=False))


if __name__=='__main__':main()
