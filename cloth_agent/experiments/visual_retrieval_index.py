"""Offline DINOv2 gallery preparation and a persistent query-only HNSW worker.

This module never constructs a camera or robot runtime. Torch/FAISS are imported
only by preparation/the worker, which can use a different Python environment.
"""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from .discovery_data import file_hash, save
from .graph_policy import checked_file, frozen, load_frozen

PROTOCOL='arm6-visual-retrieval-memory-v1'
DEFAULT_MODEL=Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small')


def graph_references(graph,trajectory,verify_images=True):
    refs={}
    for node in graph['nodes']:
        for oid in node['representative_observations']:
            if oid in refs:raise ValueError('reference belongs to multiple graph nodes: '+oid)
            record=trajectory['observations'][oid]['rgb']
            path=checked_file(record) if verify_images else Path(record['path'])
            refs[oid]=dict(image_id=oid,node_id=node['node_id'],path=str(path.resolve()),sha256=record['sha256'])
    if not refs:raise ValueError('graph has no reference RGBs')
    return [refs[k] for k in sorted(refs)]


class DinoEncoder:
    """Same checkpoint, default preprocessing and normalized FP32 CLS as benchmark."""
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
        self.calls+=1;start=time.perf_counter()
        inputs=self.processor(images=image,return_tensors='pt');prep=time.perf_counter()-start
        self.sync();start=time.perf_counter()
        with self.torch.inference_mode():
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            v=self.model(**inputs).last_hidden_state[:,0].float()
            v=self.torch.nn.functional.normalize(v,dim=-1).cpu().numpy().astype('float32')
        self.sync();return v,prep,time.perf_counter()-start


def encoder_contract(meta):
    return {k:meta[k] for k in ('weights','pooling','dtype','torch','processor')}


def build_index(vectors):
    import faiss
    faiss.omp_set_num_threads(1)
    index=faiss.IndexHNSWFlat(vectors.shape[1],32,faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction=200;index.hnsw.efSearch=64;index.add(np.ascontiguousarray(vectors,dtype='float32'))
    return index


def prepare(graph_path,output,model_path=DEFAULT_MODEL,device='cuda',reuse_memory=None):
    import faiss
    graph=load_frozen(graph_path);trajectory=load_frozen(checked_file(graph['source_trajectory']))
    refs=graph_references(graph,trajectory)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    encoder=DinoEncoder(model_path,device);cached={};reuse_provenance=None
    if reuse_memory:
        # Reuse checked vectors, not benchmark runtime/matcher code or its subset index.
        root=Path(reuse_memory);old=load_frozen(root/'memory.json')
        if encoder_contract(old['encoder'])!=encoder_contract(encoder.meta):raise ValueError('cached encoder contract differs')
        vp=root/'reference_embeddings.npy'
        if file_hash(vp)!=old['artifacts']['reference_embeddings.npy']:raise ValueError('cached vectors changed')
        vectors=np.load(vp,allow_pickle=False)
        if vectors.shape!=(len(old['images']),encoder.model.config.hidden_size) or not np.isfinite(vectors).all():
            raise ValueError('invalid cached embedding matrix')
        if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):raise ValueError('cached vectors are not normalized')
        cached={(r['image_id'],r['sha256']):v for r,v in zip(old['images'],vectors)}
        reuse_provenance=dict(memory=str((root/'memory.json').resolve()),sha256=file_hash(root/'memory.json'))
    vectors=[];reused=0
    for ref in refs:
        vector=cached.get((ref['image_id'],ref['sha256']))
        if vector is None:
            with Image.open(ref['path']) as im:vector=encoder.encode(im.convert('RGB'))[0][0]
        else:reused+=1
        vectors.append(vector)
    vectors=np.asarray(vectors,dtype='float32')
    np.save(output/'reference_embeddings.npy',vectors)
    faiss.write_index(build_index(vectors),str(output/'reference.index'))
    memory=dict(protocol=PROTOCOL,graph_path=str(Path(graph_path).resolve()),graph_sha256=file_hash(graph_path),
        trajectory=graph['source_trajectory'],images=refs,encoder=encoder.meta,dimension=vectors.shape[1],
        index=dict(type='FAISS IndexHNSWFlat',metric='normalized inner product',M=32,efConstruction=200,efSearch=64,threads=1),
        reused_embedding_count=reused,new_reference_encode_calls=encoder.calls,reused_from=reuse_provenance,
        artifacts={name:file_hash(output/name) for name in ('reference_embeddings.npy','reference.index')})
    frozen(output/'memory.json',memory)
    return dict(status='PRECOMPUTED',references=len(refs),nodes=len({r['node_id'] for r in refs}),
                reused=reused,new_encoded=encoder.calls,memory=str((output/'memory.json').resolve()))


def validate_memory(directory,graph_path=None,graph=None,trajectory=None):
    root=Path(directory);memory=load_frozen(root/'memory.json')
    if memory.get('protocol')!=PROTOCOL:raise ValueError('not a replay retrieval memory; prepare a full-graph index first')
    graph_path=Path(graph_path or memory['graph_path'])
    if file_hash(graph_path)!=memory['graph_sha256']:raise ValueError('retrieval index does not belong to the selected graph')
    graph=load_frozen(graph_path) if graph is None else graph
    trajectory=load_frozen(checked_file(graph['source_trajectory'])) if trajectory is None else trajectory
    if memory['trajectory']!=graph['source_trajectory']:raise ValueError('retrieval trajectory differs from graph')
    if graph_references(graph,trajectory)!=memory['images']:raise ValueError('retrieval gallery does not cover the complete graph')
    for name,sha in memory['artifacts'].items():
        if file_hash(root/name)!=sha:raise ValueError('retrieval artifact modified: '+name)
    return memory


class QueryIndex:
    def __init__(self,directory):
        import faiss
        self.memory=validate_memory(directory);faiss.omp_set_num_threads(1)
        self.index=faiss.read_index(str(Path(directory)/'reference.index'))
        if self.index.ntotal!=len(self.memory['images']) or self.index.d!=self.memory['dimension']:
            raise ValueError('index/gallery dimensions differ')
        self.encoder=DinoEncoder(self.memory['encoder']['path'],self.memory['encoder']['device'])
        if encoder_contract(self.encoder.meta)!=encoder_contract(self.memory['encoder']):raise ValueError('query encoder differs from reference encoder')
    def query(self,path,top_k=10,exclude_ids=()):
        if top_k<1:raise ValueError('top_k must be positive')
        started=time.perf_counter();t=started;encode_before=self.encoder.calls
        with Image.open(path) as original:image=original.convert('RGB')
        decode_s=time.perf_counter()-t
        try:vector,preprocess_s,forward_s=self.encoder.encode(image)
        finally:image.close()
        t=time.perf_counter();excluded=set(exclude_ids);refs=self.memory['images']
        k=min(top_k,len(refs)-sum(r['image_id'] in excluded for r in refs))
        if k<1:raise ValueError('no references after exclusions')
        count=min(len(refs),k+sum(r['image_id'] in excluded for r in refs))
        scores,ids=self.index.search(vector,count)
        ranking=[dict(image_id=refs[int(i)]['image_id'],node_id=refs[int(i)]['node_id'],similarity=float(score))
                 for score,i in zip(scores[0],ids[0]) if i>=0 and refs[int(i)]['image_id'] not in excluded][:k]
        return dict(ranking=ranking,query_encode_calls=self.encoder.calls-encode_before,reference_encode_calls_runtime=0,
            timing=dict(query_image_decode_s=decode_s,dino_preprocess_s=preprocess_s,dino_forward_s=forward_s,
                        ann_search_s=time.perf_counter()-t,retrieval_total_s=time.perf_counter()-started))


def serve(memory):
    # One JSON request/response at a time; library progress/warnings go to stderr.
    with contextlib.redirect_stdout(sys.stderr):index=QueryIndex(memory)
    print(json.dumps(dict(status='READY',references=index.index.ntotal,memory_sha256=file_hash(Path(memory)/'memory.json'))),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('command')=='close':return
        try:
            with contextlib.redirect_stdout(sys.stderr):result=index.query(request['image'],request.get('top_k',10),request.get('exclude_ids',[]))
            response=dict(id=request['id'],status='OK',**result)
        except Exception as exc:response=dict(id=request.get('id'),status='ERROR',error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(response,allow_nan=False),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--graph',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--model-path',type=Path,default=DEFAULT_MODEL);a.add_argument('--device',default='cuda');a.add_argument('--reuse-memory',type=Path)
    a=s.add_parser('serve');a.add_argument('--memory',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='prepare':print(json.dumps(prepare(args.graph,args.output,args.model_path,args.device,args.reuse_memory)))
    else:serve(args.memory)


if __name__=='__main__':main()
