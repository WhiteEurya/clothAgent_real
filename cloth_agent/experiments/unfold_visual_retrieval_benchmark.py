"""Offline DINOv2/HNSW retrieval and unchanged GPT cloth-matcher benchmark.

No robot runtime is constructed. Reference vectors and indexes are prepared in a
separate command. Timed runs encode queries only; graph groups are proxy labels.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema
import numpy as np
from PIL import Image
from .discovery_data import save, read, file_hash, canonical, digest
from .graph_policy import load_frozen, frozen, checked_file, Requests
from .graph_policy_matching import topology_match, signatures, validate_match_response
from .graph_policy_schema import MATCH
from .arm6_traj_replay import GARMENT_FRAME, matching_graph
from .graph_policy_replay import PROMPTS
from . import discovery_model

VERSION='unfold-visual-retrieval-benchmark-v1'
DEFAULT_MODEL=Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small')
DEFAULT_GRAPH=Path('reports/arm6_archive_graph_20261019/merged/graph.json')
STAGES=['query_image_decode_s','dino_preprocess_s','dino_forward_s','ann_search_s',
        'candidate_loading_s','gpt_rerank_s','other_host_s']
MODES=['retrieval_only','retrieval_gpt_rerank','current_global_match']
SIZES=[100,500,1000,5000,10000]
COMMON=('You match cloth observations and ground an existing Arm6 demonstration. Images are attached in image_ids order. '
        'Return only schema-valid JSON. Do not generate robot code.\n'+GARMENT_FRAME)
RUNTIME_SOURCES=['graph_policy_matching.py','graph_policy_replay.py','arm6_traj_replay.py','discovery_model.py','graph_policy_schema.py','graph_policy.py']


def decode(path):
    with Image.open(path) as im:return im.convert('RGB')


def pixel_hash(path):
    with Image.open(path) as im:
        im=im.convert('RGB');return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()


def source_hashes():
    base=Path(__file__).parent
    return {name:file_hash(base/name) for name in RUNTIME_SOURCES}


def dataset_from_graph(graph_path,session_prefix='20261019',sessions=()):
    graph=load_frozen(graph_path);trajectory=load_frozen(checked_file(graph['source_trajectory']))
    selected=[s for s in trajectory['steps'] if
              (s['session'] in sessions if sessions else s['session'].startswith(session_prefix))]
    if sessions and set(sessions)-{s['session'] for s in selected}:raise ValueError('unknown requested sessions')
    if not selected:raise ValueError('no selected unfolding trajectories')
    obsids={s[key] for s in selected for key in ('before_observation','after_observation') if s[key]}
    labels={o:n['node_id'] for n in graph['nodes'] for o in n['representative_observations']}
    ownership=defaultdict(list)
    for s in selected:
        manifest=read(checked_file(s['commanded_action']['manifest']))
        if manifest.get('status')!='COMPLETE' or manifest.get('operator_accepted') is not True:
            raise ValueError('requires operator-confirmed complete demonstrations')
        for key in ('before_observation','after_observation'):
            if s[key]:ownership[s[key]].append(dict(session=s['session'],iteration=s['iteration'],when=key))
    images=[]
    for oid in sorted(obsids):
        ref=trajectory['observations'][oid]['rgb'];path=checked_file(ref)
        if oid not in labels:raise ValueError('observation missing proxy graph group')
        images.append(dict(image_id=oid,path=str(path.resolve()),sha256=ref['sha256'],
            pixel_sha256=pixel_hash(path),proxy_group=labels[oid],sources=ownership[oid]))
    return dict(version=VERSION,images=images,graph_path=str(Path(graph_path).resolve()),graph_sha256=file_hash(graph_path),
        trajectory_path=graph['source_trajectory']['path'],trajectory_sha256=graph['source_trajectory']['sha256'],
        sessions=sorted({s['session'] for s in selected}),session_selection='explicit sessions' if sessions else f'session prefix {session_prefix}',
        trajectory_count=len(selected),label_type='PROXY_GRAPH_GROUP_NOT_HUMAN_VERIFIED',runtime_sources=source_hashes())


def positives(images,query,cross_session=False):
    query_sessions={s['session'] for s in query['sources']}
    return {r['image_id'] for r in images if r['proxy_group']==query['proxy_group']
            and r['image_id']!=query['image_id'] and r['pixel_sha256']!=query['pixel_sha256']
            and (not cross_session or query_sessions.isdisjoint(s['session'] for s in r['sources']))}


def allowed_references(images,query):
    return [r for r in images if r['image_id']!=query['image_id'] and r['pixel_sha256']!=query['pixel_sha256']]


def stable_query_schedule(images,repeats,seed):
    # Fixed paired schedule, spread across proxy groups rather than only easy positives.
    groups=defaultdict(list)
    for r in images:groups[r['proxy_group']].append(r['image_id'])
    rng=np.random.default_rng(seed);keys=sorted(groups);rng.shuffle(keys)
    for values in groups.values():rng.shuffle(values)
    order=[]
    while any(groups.values()):
        for key in keys:
            if groups[key]:order.append(groups[key].pop())
    return [order[i%len(order)] for i in range(repeats)]


class DinoEncoder:
    def __init__(self,path,device='cuda'):
        import torch
        from transformers import AutoModel,AutoImageProcessor
        self.torch=torch;self.device=device;self.calls=0
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True).eval().to(device).requires_grad_(False)
        self.processor=AutoImageProcessor.from_pretrained(str(path),local_files_only=True,use_fast=False)
        if self.model.config.model_type!='dinov2':raise ValueError('DINOv2 weights required')
        self.meta=dict(model='facebook/dinov2-small',path=str(Path(path).resolve()),
            weights={p.name:file_hash(p) for p in sorted(Path(path).iterdir()) if p.is_file()},
            pooling='last_hidden_state[:,0] CLS, L2 normalized',dtype='float32',device=device,
            torch=torch.__version__,processor=self.processor.to_dict(),
            gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None)
    def sync(self):
        if self.device.startswith('cuda'):self.torch.cuda.synchronize()
    def encode(self,image):
        self.calls+=1;t=time.perf_counter()
        inputs=self.processor(images=image,return_tensors='pt')
        prep=time.perf_counter()-t;self.sync();t=time.perf_counter()
        with self.torch.inference_mode():
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            output=self.model(**inputs).last_hidden_state[:,0].float()
            output=self.torch.nn.functional.normalize(output,dim=-1)
            vector=output.cpu().numpy().astype('float32')
        self.sync();return vector,prep,time.perf_counter()-t


def build_index(vectors):
    import faiss
    faiss.omp_set_num_threads(1)
    index=faiss.IndexHNSWFlat(vectors.shape[1],32,faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction=200;index.hnsw.efSearch=64
    index.add(np.ascontiguousarray(vectors,dtype='float32'))
    return index


def prepare(graph_path,output,model_path=DEFAULT_MODEL,device='cuda',session_prefix='20261019',sessions=()):
    import faiss
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dataset=dataset_from_graph(graph_path,session_prefix,sessions)
    encoder=DinoEncoder(model_path,device);vectors=[];started=time.perf_counter()
    for i,ref in enumerate(dataset['images']):
        image=decode(ref['path']);vector,_,_=encoder.encode(image);image.close();vectors.append(vector[0])
        if (i+1)%20==0:print(f'Precompute references: {i+1}/{len(dataset["images"])}',flush=True)
    vectors=np.ascontiguousarray(vectors,dtype='float32');np.save(output/'reference_embeddings.npy',vectors)
    faiss.write_index(build_index(vectors),str(output/'reference.index'))
    scaling=output/'scaling';scaling.mkdir()
    for size in SIZES:
        ids=np.arange(size)%len(vectors);scaled=vectors[ids]
        faiss.write_index(build_index(scaled),str(scaling/f'{size}.index'))
    dataset.update(encoder=encoder.meta,index=dict(type='FAISS IndexHNSWFlat',metric='normalized inner product',M=32,
        efConstruction=200,efSearch=64,threads=1,faiss_version=faiss.__version__),
        reference_encode_calls=encoder.calls,precompute_s=time.perf_counter()-started,
        artifacts={str(p.relative_to(output)):file_hash(p) for p in [output/'reference_embeddings.npy',output/'reference.index',*sorted(scaling.glob('*.index'))]},
        scaling_note='Embedding rows repeated cyclically when needed; latency-only indexes, never accuracy evidence.')
    frozen(output/'memory.json',dataset)
    print(json.dumps(dict(status='PRECOMPUTED',images=len(vectors),trajectory_count=dataset['trajectory_count'],
        proxy_groups=len({r['proxy_group'] for r in dataset['images']}),output=str(output)),ensure_ascii=False),flush=True)
    return dataset


def load_memory(path):
    import faiss
    path=Path(path);data=load_frozen(path/'memory.json')
    for name,sha in data['artifacts'].items():
        if file_hash(path/name)!=sha:raise ValueError('precomputed artifact changed: '+name)
    for r in data['images']:
        if file_hash(r['path'])!=r['sha256']:raise ValueError('reference RGB changed')
    if source_hashes()!=data['runtime_sources']:raise ValueError('matcher source changed since memory preparation')
    faiss.omp_set_num_threads(1)
    vectors=np.load(path/'reference_embeddings.npy',mmap_mode='r')
    index=faiss.read_index(str(path/'reference.index'))
    if index.ntotal!=len(data['images']) or vectors.shape[0]!=index.ntotal:raise ValueError('index/gallery mismatch')
    return data,index


def retrieve(index,vector,images,query,k=20):
    # Oversample for self/content duplicates, increase only if exclusion needs it.
    excluded=sum(r['pixel_sha256']==query['pixel_sha256'] or r['image_id']==query['image_id'] for r in images)
    count=min(index.ntotal,k+excluded)
    scores,ids=index.search(vector,count)
    rows=[]
    for score,i in zip(scores[0],ids[0]):
        if i<0:continue
        ref=images[int(i)]
        if ref['image_id']==query['image_id'] or ref['pixel_sha256']==query['pixel_sha256']:continue
        rows.append(dict(image_id=ref['image_id'],similarity=float(score)))
        if len(rows)==k:break
    return rows


class TimedImage:
    """Observe existing transport decode/PNG-copy time without changing image bytes."""
    def __init__(self,image,backend,is_query):self.image=image;self.backend=backend;self.is_query=is_query
    def __enter__(self):self.image.__enter__();return self
    def __exit__(self,*a):return self.image.__exit__(*a)
    def convert(self,*a,**kw):
        t=time.perf_counter();value=self.image.convert(*a,**kw)
        self.backend.add_image_time(time.perf_counter()-t,self.is_query)
        return TimedImage(value,self.backend,self.is_query)
    def save(self,*a,**kw):
        t=time.perf_counter()
        try:return self.image.save(*a,**kw)
        finally:self.backend.add_image_time(time.perf_counter()-t,self.is_query)


class TimedBackend(discovery_model.CodexMultimodal):
    def reset(self):self.reference_loading_s=0.;self.query_transport_s=0.;self.model_wall_s=0.;self.calls=0
    def add_image_time(self,elapsed,is_query):
        if is_query:self.query_transport_s+=elapsed
        else:self.reference_loading_s+=elapsed
    def invoke(self,prompt,images,schema):
        # Patch only this module's PIL handle, in this single-threaded benchmark.
        # The original invoke, model, reasoning effort, isolation and PNG copies run unchanged.
        before_ref=self.reference_loading_s;before_query=self.query_transport_s
        counter=[0]
        def image_open(*a,**kw):
            is_query=counter[0]==0;counter[0]+=1;t=time.perf_counter();im=Image.open(*a,**kw)
            self.add_image_time(time.perf_counter()-t,is_query)
            return TimedImage(im,self,is_query)
        self.calls+=1;t=time.perf_counter()
        try:
            with patch.object(discovery_model,'Image',SimpleNamespace(open=image_open)):
                return super().invoke(prompt,images,schema)
        finally:
            self.model_wall_s+=time.perf_counter()-t-(self.reference_loading_s-before_ref)-(self.query_transport_s-before_query)


class SingleShotRequests(Requests):
    """Same matcher prompt/schema; benchmark Top-10 gets exactly one model invocation."""
    def ask(self,kind,payload,images,validator=None):
        schema=self.schemas[kind]
        prompt=self.common+'\n'+self.prompts[kind]+'\n'+canonical(dict(payload=payload,image_ids=[k for k,_ in images],schema=schema))
        self.output.mkdir(parents=True,exist_ok=True)
        save(self.output/'request.json',dict(prompt=prompt,images=[dict(id=k,path=str(p),sha256=file_hash(p)) for k,p in images],
            model=self.backend.model,cache_enabled=False,retries=0))
        response=self.backend.invoke(prompt,[p for _,p in images],schema)
        save(self.output/'raw_response.json',response)
        value=json.loads(response['text']);jsonschema.validate(value,schema)
        if validator:validator(value)
        save(self.output/'response.json',value);return value


def filtered_graph(graph,images,query,candidate_ids=None,individual_candidates=False):
    allowed={r['image_id'] for r in allowed_references(images,query)}
    if candidate_ids is not None:allowed&=set(candidate_ids)
    nodes=[]
    by_ref={o:n for n in graph['nodes'] for o in n['representative_observations']}
    if individual_candidates:
        for oid in candidate_ids:
            if oid not in allowed:raise ValueError('candidate is self or absent')
            node=copy.deepcopy(by_ref[oid]);node['node_id']=oid;node['representative_observations']=[oid];nodes.append(node)
    else:
        for original in graph['nodes']:
            refs=[o for o in original['representative_observations'] if o in allowed]
            if refs:
                node=copy.deepcopy(original);node['representative_observations']=refs;nodes.append(node)
    return matching_graph(dict(graph,nodes=nodes,edges=[]))


def trial(mode,query,memory,index,encoder,graph,trajectory,backend,folder):
    row={stage:0. for stage in STAGES};row.update(mode=mode,query_id=query['image_id'],
        index_size=len(memory['images']),status='OK',error='',candidate_count=0,model_calls=0,ranking=[],match={})
    backend.reset();started=time.perf_counter();encode_before=encoder.calls
    try:
        t=time.perf_counter();image=decode(query['path']);row['query_image_decode_s']=time.perf_counter()-t
        if mode!='current_global_match':
            vector,row['dino_preprocess_s'],row['dino_forward_s']=encoder.encode(image)
            t=time.perf_counter();row['ranking']=retrieve(index,vector,memory['images'],query,10 if mode=='retrieval_gpt_rerank' else 20)
            row['ann_search_s']=time.perf_counter()-t
        image.close()
        if mode=='retrieval_gpt_rerank':
            candidate_ids=[r['image_id'] for r in row['ranking'][:10]]
            if len(candidate_ids)!=10:raise ValueError('Top-10 requires ten non-self reference images')
            t=time.perf_counter();subset=filtered_graph(graph,memory['images'],query,candidate_ids,True)
            lookup={r['image_id']:r for r in memory['images']}
            images=[('current',Path(query['path']))]+[(oid,Path(lookup[oid]['path'])) for oid in candidate_ids]
            row['candidate_count']=len(candidate_ids);row['candidate_loading_s']+=time.perf_counter()-t
            req=SingleShotRequests(folder/'model',backend,schemas={'node_match':MATCH},prompts={'node_match':PROMPTS['node_match']},common=COMMON,no_cache=True,max_calls=1)
            value=req.ask('node_match',signatures(subset,candidate_ids,with_representatives=True),images,
                validator=lambda v:validate_match_response(v,dict(nodes=subset['match_nodes']),.8))
            row['match']=dict(status='UNKNOWN' if value['status']=='NO_MATCH' else value['status'],
                best_candidate=value['node_id'] or None,original_response=value)
        elif mode=='current_global_match':
            t=time.perf_counter();subset=filtered_graph(graph,memory['images'],query)
            row['candidate_count']=sum(len(n['representative_observations']) for n in subset['match_nodes'])
            row['candidate_loading_s']+=time.perf_counter()-t
            req=Requests(folder/'model',backend,schemas={'node_match':MATCH},prompts={'node_match':PROMPTS['node_match']},
                common=COMMON,no_cache=True,max_calls=2)
            value=topology_match(subset,trajectory,SimpleNamespace(image=Path(query['path'])),req,folder/'matching',{},pending=None,threshold=.8)
            row['match']=value
    except Exception as exc:
        row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
    finally:
        row['total_s']=time.perf_counter()-started
        row['candidate_loading_s']+=backend.reference_loading_s
        row['gpt_rerank_s']=backend.model_wall_s
        row['query_transport_s']=backend.query_transport_s
        row['model_calls']=backend.calls
        row['query_encode_calls']=encoder.calls-encode_before
        row['other_host_s']=max(0.,row['total_s']-sum(row[s] for s in STAGES if s!='other_host_s'))
    expected=0 if mode=='current_global_match' else 1
    if row['query_encode_calls']>expected or (row['status']=='OK' and row['query_encode_calls']!=expected):
        raise RuntimeError('unexpected encoding count; references must never be encoded during a trial')
    return row


def statistics(values):
    x=np.asarray(values,float)
    return dict(n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),p95=float(np.percentile(x,95))) if len(x) else dict(n=0,median=None,mean=None,p95=None)


def append_row(output,row):
    with (output/'rows.jsonl').open('a') as f:f.write(canonical(row)+'\n')


def accuracy(memory,index,encoder,output):
    rows=[];images=memory['images'];vectors=[]
    # Re-encode each RGB as a query; do not substitute its gallery embedding.
    for q in images:
        image=decode(q['path']);v,_,_=encoder.encode(image);image.close();vectors.append(v[0])
        ranking=retrieve(index,v,images,q,20);ids=[r['image_id'] for r in ranking]
        pos=positives(images,q);cross=positives(images,q,True)
        rows.append(dict(query_id=q['image_id'],proxy_group=q['proxy_group'],positive_ids=sorted(pos),
            cross_session_positive_ids=sorted(cross),ranking=ranking,
            hit={str(k):bool(pos&set(ids[:k])) if pos else None for k in (1,5,10,20)},
            fraction={str(k):len(pos&set(ids[:k]))/len(pos) if pos else None for k in (1,5,10,20)},
            cross_session_hit={str(k):bool(cross&set(ids[:k])) if cross else None for k in (1,5,10,20)}))
    save(output/'proxy_recall.json',dict(label_type='PROXY_GRAPH_GROUP_NOT_HUMAN_VERIFIED',rows=rows))
    np.save(output/'accuracy_query_vectors.npy',np.asarray(vectors,dtype='float32'))
    return rows


def run(memory_path,output,*,repeats=20,warmup=1,seed=20261019,model='gpt-6-astra',timeout=600.,modes=MODES,resume=False):
    if repeats<20 or warmup<1:raise ValueError('at least 20 measured trials and one warmup per condition required')
    if set(modes)-set(MODES) or len(set(modes))!=len(modes):raise ValueError('invalid modes')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    memory,index=load_memory(memory_path)
    graph=load_frozen(memory['graph_path']);trajectory=load_frozen(memory['trajectory_path'])
    if file_hash(memory['graph_path'])!=memory['graph_sha256']:raise ValueError('proxy graph changed')
    if file_hash(memory['trajectory_path'])!=memory['trajectory_sha256']:raise ValueError('trajectory archive changed')
    encoder=DinoEncoder(memory['encoder']['path'],memory['encoder']['device'])
    if encoder.meta!=memory['encoder']:raise ValueError('query encoder differs from precomputed gallery encoder')
    backend=TimedBackend(model,timeout);backend.reset()
    schedule=stable_query_schedule(memory['images'],repeats,seed);lookup={r['image_id']:r for r in memory['images']}
    config=dict(version=VERSION,memory=str(Path(memory_path).resolve()),memory_sha256=file_hash(Path(memory_path)/'memory.json'),
        repeats=repeats,warmup=warmup,seed=seed,modes=modes,query_schedule=schedule,model=model,reasoning_effort='high (existing default)',
        model_timeout_s=timeout,model_cache=False,concurrency=1,python=platform.python_version(),host=platform.platform(),
        timing_note='Wall time, CUDA synchronized. Candidate loading includes unchanged matcher reference RGB decode and PNG transport copies. GPT stage includes CLI/isolation/network/validation transport overhead, not pure server inference. Query transport copying is in other_host.',
        baseline_note='Existing topology_match global path, same Arm6 common prompt and node_match prompt/schema/validator, no-cache Requests and existing bounded repair. Reference universe restricted to same unfolding memory and query excluded. Rerank uses identical matcher with each reference image as a candidate node, single-shot; NO_MATCH mapped to UNKNOWN.',
        labels='proxy graph groups; no human accuracy claim',runtime_sources=source_hashes())
    frozen(output/'config.json',config)
    if not (output/'proxy_recall.json').exists():accuracy(memory,index,encoder,output)
    done=set()
    if (output/'rows.jsonl').exists():
        done={(r['condition'],r['iteration']) for r in map(json.loads,(output/'rows.jsonl').read_text().splitlines())}
    # Re-warm on every new process, including resumed runs. Warmups are separately saved.
    for mode in modes:
        for w in range(warmup):
            print(f'WARMUP {mode} {w+1}/{warmup}',flush=True)
            row=trial(mode,lookup[schedule[w%len(schedule)]],memory,index,encoder,graph,trajectory,backend,output/'warmup'/mode/str(time.time_ns()))
            with (output/'warmup.jsonl').open('a') as f:f.write(canonical(row)+'\n')
            print(f'  {row["status"]}: {row["total_s"]:.3f}s',flush=True)
    # Rotate mode order within each paired query to reduce time-of-day/order bias.
    for i,qid in enumerate(schedule):
        order=modes[i%len(modes):]+modes[:i%len(modes)]
        for mode in order:
            if (mode,i) in done:continue
            folder=output/'trials'/mode/f'{i:03d}';folder.mkdir(parents=True,exist_ok=True)
            print(f'TRIAL {i+1}/{repeats} {mode} query={qid}',flush=True)
            row=trial(mode,lookup[qid],memory,index,encoder,graph,trajectory,backend,folder)
            row.update(condition=mode,iteration=i,synthetic_index=False)
            save(folder/'trial.json',row);append_row(output,row)
            report(output)
            print(f'  {row["status"]}: total={row["total_s"]:.3f}s GPT={row["gpt_rerank_s"]:.3f}s',flush=True)
    scaling(memory_path,output,repeats,warmup)
    if source_hashes()!=config['runtime_sources']:raise RuntimeError('Graph matcher files were modified during benchmark')
    report(output)


def scaling(memory_path,output,repeats,warmup):
    import faiss
    vectors=np.load(output/'accuracy_query_vectors.npy')
    done={(r['condition'],r['iteration']) for r in map(json.loads,(output/'rows.jsonl').read_text().splitlines())}
    for size in SIZES:
        index=faiss.read_index(str(Path(memory_path)/'scaling'/f'{size}.index'))
        for i in range(warmup):index.search(vectors[i%len(vectors)][None],20)
        for i in range(repeats):
            condition=f'ann_scaling_{size}'
            if (condition,i) in done:continue
            vector=np.ascontiguousarray(vectors[i%len(vectors)][None]);t=time.perf_counter();index.search(vector,20);elapsed=time.perf_counter()-t
            row={s:0. for s in STAGES};row.update(mode='ann_scaling',condition=condition,iteration=i,index_size=size,
                query_id=f'cached_query_vector_{i%len(vectors)}',status='OK',error='',candidate_count=20,model_calls=0,
                query_encode_calls=0,synthetic_index=True,ann_search_s=elapsed,total_s=elapsed)
            append_row(output,row)


def report(output):
    output=Path(output)
    fmt=lambda x:'NA' if x is None else f'{x:.6f}'
    rows=[json.loads(s) for s in (output/'rows.jsonl').read_text().splitlines()] if (output/'rows.jsonl').exists() else []
    config=read(output/'config.json');recall=read(output/'proxy_recall.json').get('rows',[])
    fields=['condition','mode','iteration','query_id','index_size','candidate_count','status','error','synthetic_index',
            'model_calls','query_encode_calls',*STAGES,'query_transport_s','total_s']
    with (output/'benchmark.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary={}
    for condition in dict.fromkeys(r['condition'] for r in rows):
        group=[r for r in rows if r['condition']==condition];ok=[r for r in group if r['status']=='OK']
        summary[condition]=dict(attempts=len(group),successes=len(ok),failures=len(group)-len(ok),
            stages={stage:statistics([r[stage] for r in ok]) for stage in [*STAGES,'total_s']},
            all_attempt_stages={stage:statistics([r[stage] for r in group]) for stage in [*STAGES,'total_s']},
            all_attempt_total=statistics([r['total_s'] for r in group]))
    save(output/'statistics.json',summary)
    early=read(output/'early_stop.json') if (output/'early_stop.json').exists() else None
    paired=set.intersection(*[{r['iteration'] for r in rows if r['mode']==m} for m in config.get('modes',MODES)])
    paired_success=set.intersection(*[{r['iteration'] for r in rows if r['mode']==m and r['status']=='OK'} for m in config.get('modes',MODES)])
    paired_summary={m:{stage:statistics([r[stage] for r in rows if r['mode']==m and r['iteration'] in paired_success])
        for stage in [*STAGES,'total_s']} for m in config.get('modes',MODES)}
    save(output/'paired_success_statistics.json',dict(iterations=sorted(paired_success),conditions=paired_summary))
    early_in_effect=bool(early and len(paired)<=early['stop_after_pairs'])
    lines=['# unfold_visual_retrieval_benchmark','',
        'Offline only. Graph runtime and matcher source files are unchanged. No robot connection.',
        '']
    if early:
        lines += ['## Exploratory benchmark / early-stopping disclosure','',
            f'**Original prespecified plan: {config["repeats"]} paired samples per mode. Completed paired attempts: {len(paired)}; pairs with successful responses in every mode: {len(paired_success)}.**',
            f'The operator requested early stopping at {early["stop_after_pairs"]} complete paired samples after observing a stable large latency effect. '
            'This was an outcome-dependent decision made during the experiment; **n=10 was not the prespecified experimental design**.',
            'Results are exploratory descriptive statistics, not a confirmatory result. Outcome-dependent stopping can bias effect-size estimates; '
            'the p95 estimate with ten samples is especially uncertain. No formal sequential stopping rule or significance test was specified.',
            'The original 20-repeat config, query schedule, warmups, raw trials and original benchmark source are preserved for later supplementation. '
            'ANN scaling still uses 20 low-cost measurements at each index size.',
            '' if early_in_effect else ('The original main-mode sample target has since been reached; the earlier outcome-dependent stopping decision remains disclosed.'
                if len(paired)>=config['repeats'] else 'Supplementation beyond the early-stop point is in progress; the original 20-pair target has not yet been reached.'),
            '']
    lines += ['## Failure accounting','',
        f'There are {len(paired)} completed paired attempts and {len(paired_success)} all-successful pairs. '
        'The primary latency tables include EVERY attempt and measure time to a terminal result, including errors. '
        'A failed request is not a successful match and is not silently discarded or replaced. '
        'A separate comparison below restricts all modes to the same all-successful query pairs.', '']
    for row in rows:
        if row['status']!='OK':lines.append(f'- Failure: `{row["condition"]}`, iteration {row["iteration"]}, query `{row.get("query_id")}`, {row["total_s"]:.3f} s: {row["error"]}')
    lines+=['','## Method','',
        '- References are pre-encoded DINOv2-small CLS vectors, float32 and L2-normalized; FAISS HNSW inner product.',
        '- Same query schedule across modes; one query at a time, interleaved mode order. Model/result cache disabled.',
        f'- Warmup: {config.get("warmup")} complete trials per mode per process, excluded. Original planned target: {config.get("repeats")} measured trials per condition; actual counts are reported below.',
        '- Query ID and exact RGB-pixel duplicates are excluded in ALL real-data modes; same-state adjacent captures may remain.',
        '- No fine-tuning, crop tuning, precision changes, matcher prompt optimization or action planning.',
        '- GPT time is observed client wall time including existing CLI/isolation/network overhead, not server-only inference.',
        '- Candidate loading measures the existing reference RGB decode and PNG transport copies; residual hashing, query transport and JSON/log overhead are in other_host.',
        '- DINO forward includes tensor transfer, forward, CLS normalization and CPU vector copy, with CUDA synchronization; CPU preprocessing is separate.',
        '- GPU clocks are not locked. Interleaved long GPT calls leave variable GPU idle gaps, so DINO timings include these operating conditions despite initial warmup and resident model weights.',
        '- Baseline uses the existing global topology_match entrypoint on the SAME unfolding reference pool (not unrelated folding images). Existing repair calls are counted; rerank submits query + Top-10 once, without repair.',
        '- Rerank preserves current node_match prompt/schema: candidate image IDs become singleton node IDs; NO_MATCH is exposed as UNKNOWN. AMBIGUOUS/UNKNOWN has no invented best candidate.',
        '','## End-to-end attempt latency (seconds; includes failures)','',
        '| Condition | n attempts | Successful | Failed | Median | Mean | p95 | Median speedup vs global |','|---|---:|---:|---:|---:|---:|---:|---:|']
    baseline=summary.get('current_global_match',{}).get('all_attempt_total',{}).get('median')
    for name,s in summary.items():
        v=s['all_attempt_total']
        speed=baseline/v['median'] if baseline and v['median'] and name in MODES else None
        lines.append(f'| {name} | {s["attempts"]} | {s["successes"]} | {s["failures"]} | {fmt(v["median"])} | {fmt(v["mean"])} | {fmt(v["p95"])} | {fmt(speed)} |')
    lines+=['','## Successful paired comparison (same queries in every mode)','',
        f'Eligible iteration IDs: {sorted(paired_success)}. This success-conditioned subset is supplementary, not a replacement for the n={len(paired)} attempted dataset.',
        '', '| Condition | Successful paired n | Median | Mean | p95 | Median speedup vs global |','|---|---:|---:|---:|---:|---:|']
    paired_baseline=paired_summary.get('current_global_match',{}).get('total_s',{}).get('median')
    for name,s in paired_summary.items():
        v=s['total_s'];speed=paired_baseline/v['median'] if paired_baseline and v['median'] else None
        lines.append(f'| {name} | {v["n"]} | {fmt(v["median"])} | {fmt(v["mean"])} | {fmt(v["p95"])} | {fmt(speed)} |')
    lines+=['','## Stage statistics (seconds; all attempts, including failures)','', '| Condition | Stage | n | Median | Mean | p95 |','|---|---|---:|---:|---:|---:|']
    for name,s in summary.items():
        if name not in MODES:continue
        for stage,v in s['all_attempt_stages'].items():lines.append(f'| {name} | {stage} | {v["n"]} | {fmt(v["median"])} | {fmt(v["mean"])} | {fmt(v["p95"])} |')
    lines+=['','## Proxy recall on actual trajectory RGB','',
        'Labels are existing model-compiled graph groups, explicitly **not human-verified ground truth**. Recall@K below is the image-retrieval convention: fraction of eligible queries with at least one other same-group reference in Top-K. Fractional relevant-set recall is also reported.',
        '',f'Total queries: {len(recall)}; eligible: {sum(bool(r["positive_ids"]) for r in recall)}; no non-self positive: {sum(not r["positive_ids"] for r in recall)} (excluded from recall denominator).',
        '', '| K | Proxy Recall@K (hit rate) | Mean fraction of positives retrieved | Cross-session-positive hit rate | Cross-session eligible n |','|---|---:|---:|---:|---:|']
    recall_summary={}
    for k in (1,5,10,20):
        hit=[r['hit'][str(k)] for r in recall if r['hit'][str(k)] is not None]
        fraction=[r['fraction'][str(k)] for r in recall if r['fraction'][str(k)] is not None]
        cross=[r['cross_session_hit'][str(k)] for r in recall if r['cross_session_hit'][str(k)] is not None]
        values=[float(np.mean(v)) if v else None for v in (hit,fraction,cross)]
        recall_summary[str(k)]=dict(hit_rate=values[0],fractional_recall=values[1],cross_session_hit_rate=values[2],cross_session_eligible=len(cross))
        lines.append(f'| {k} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {len(cross)} |')
    save(output/'recall_summary.json',recall_summary)
    lines+=['','## Limits and completeness','',
        '- Scaling indexes of 100/500/1000/5000/10000 repeat reference vectors as needed, with no jitter. These measure ANN latency only, not accuracy or realistic diversity scaling.',
        '- Scaling times search precomputed query vectors; they are not full query-image pipeline times.',
        '- Proxy labels can be incorrect and were derived from these images. Adjacent before/after captures can be nearly identical; high proxy recall does not establish cross-run generalization.',
        '- Warm filesystem caches are allowed; disk caches are not artificially cleared. Provider-side prompt caching is not controllable by this benchmark.',
        '- Failures/timeouts are retained in benchmark.csv and included in primary time-to-terminal statistics. Successful-only statistics and paired-success statistics are separately retained. No failure is silently replaced.',
        '- This is a paired set of varied queries, not 20 repeats of each individual image. Raw per-query rows are retained.',
        '']
    main_target=early['stop_after_pairs'] if early_in_effect else config['repeats']
    missing=[m for m in config.get('modes',[]) if summary.get(m,{}).get('attempts',0)<main_target]
    missing += [f'ann_scaling_{s}' for s in SIZES if summary.get(f'ann_scaling_{s}',{}).get('attempts',0)<config['repeats']]
    completion=('IN PROGRESS: '+', '.join(missing) if missing else
        f'EXPLORATORY EARLY STOP: {len(paired)} completed paired attempts, {len(paired_success)} all-successful pairs; all five ANN scaling conditions complete (20 each). Original planned target remains 20.' if early_in_effect else
        'all requested main-mode and scaling trials recorded')
    lines.append('Run completeness: '+completion+'.')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    if rows:plot(output,summary)


def plot(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[n for n in MODES if n in summary and summary[n]['attempts']]
    if names:
        fig,axes=plt.subplots(1,3,figsize=(20,6),layout='constrained');bottom=np.zeros(len(names))
        early=read(output/'early_stop.json') if (output/'early_stop.json').exists() else None
        if early:
            counts='/'.join(str(summary[n]['attempts']) for n in names)
            successes='/'.join(str(summary[n]['successes']) for n in names)
            fig.suptitle(f'EXPLORATORY: attempts n={counts}, successful={successes}; planned n=20 per mode\nOutcome-dependent early stop after observing a stable large effect (not prespecified n=10)',fontsize=13)
        for stage in STAGES:
            values=np.array([summary[n]['all_attempt_stages'][stage]['mean'] for n in names])
            axes[0].bar(names,values,bottom=bottom,label=stage.removesuffix('_s'));bottom+=values
        axes[0].set_ylabel('Seconds (mean stage contributions)');axes[0].tick_params(axis='x',labelrotation=15)
        axes[0].legend(fontsize=8);axes[0].set_title('All attempts: stage means (including errors)')
        x=np.arange(len(names))
        for j,metric in enumerate(('median','mean','p95')):
            axes[1].bar(x+(j-1)*.25,[summary[n]['all_attempt_total'][metric] for n in names],.25,label=metric)
        axes[1].set_xticks(x,names,rotation=15);axes[1].set_yscale('log');axes[1].set_ylabel('Time to terminal result, seconds (log scale)');axes[1].legend()
        if 'retrieval_only' in names:
            stages=[s for s in STAGES if summary['retrieval_only']['stages'][s]['mean']>0]
            axes[2].barh([s.removesuffix('_s') for s in stages],
                [summary['retrieval_only']['stages'][s]['mean']*1000 for s in stages])
            axes[2].set_xlabel('Milliseconds (mean)');axes[2].set_title('Retrieval-only stages, enlarged')
        fig.savefig(output/'latency_breakdown.png',dpi=160);fig.savefig(output/'latency_breakdown.svg');plt.close(fig)
    sizes=[s for s in SIZES if f'ann_scaling_{s}' in summary]
    if sizes:
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        for metric in ('median','mean','p95'):
            ax.plot(sizes,[summary[f'ann_scaling_{s}']['stages']['ann_search_s'][metric]*1000 for s in sizes],marker='o',label=metric)
        ax.set_xscale('log');ax.set_xlabel('Index vectors (cyclic duplicates; LATENCY ONLY)');ax.set_ylabel('ANN search milliseconds');ax.legend();ax.grid(alpha=.3)
        fig.savefig(output/'ann_scaling.png',dpi=160);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--graph',type=Path,default=DEFAULT_GRAPH);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-path',type=Path,default=DEFAULT_MODEL);p.add_argument('--device',default='cuda')
    p.add_argument('--session-prefix',default='20261019');p.add_argument('--session',action='append',default=[])
    p=sub.add_parser('run');p.add_argument('--memory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmup',type=int,default=1);p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--timeout-s',type=float,default=600.)
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('report');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare':prepare(args.graph,args.output,args.model_path,args.device,args.session_prefix,args.session)
    elif args.command=='run':run(args.memory,args.output,repeats=args.repeats,warmup=args.warmup,seed=args.seed,
        model=args.model,timeout=args.timeout_s,modes=args.modes,resume=args.resume)
    else:report(args.output)


if __name__=='__main__':main()
