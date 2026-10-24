"""Independent DINO Top-K + one GPT state-matching request for Arm6 replay.

No action selection/planning; uncertain results remain uncertain. The gallery is
prepared offline. A persistent local worker keeps ML dependencies out of cali.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import jsonschema
from .discovery_data import canonical, digest, file_hash, save
from .graph_policy import checked_file, evidence_file, load_frozen
from .graph_policy_matching import signatures, validate_match_response
from .visual_retrieval_index import validate_memory

ROOT=Path(__file__).resolve().parents[2]


def worker_python(explicit=None):
    # Resolving a venv's Python symlink would lose that venv's site-packages.
    if explicit:return str(Path(explicit).expanduser().absolute())
    if all(importlib.util.find_spec(m) is not None for m in ('torch','transformers','faiss')):return sys.executable
    path=Path.home()/'.cache/cloth_agent/visual_retrieval_env/bin/python'
    if path.is_file():return str(path)
    raise ValueError('retrieval environment missing; pass --retrieval-python with torch, transformers and faiss installed')


class RetrievalWorker:
    def __init__(self,memory,output,python=None,timeout=120.):
        self.process=None;self.selector=None;self.log=None;self.buffer=b'';self.sequence=0;self.timeout=timeout
        if timeout<=0:raise ValueError('worker timeout must be positive')
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        try:
            self.log=(output/'retrieval_worker.stderr.log').open('ab')
            self.process=subprocess.Popen([worker_python(python),'-u','-m','cloth_agent.experiments.visual_retrieval_index',
                'serve','--memory',str(Path(memory).resolve())],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
            self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ)
            self.ready=self._receive()
            if self.ready.get('status')!='READY':raise RuntimeError('retrieval worker did not initialize')
            if self.ready.get('memory_sha256')!=file_hash(Path(memory)/'memory.json'):raise ValueError('worker loaded different retrieval memory')
        except BaseException:
            self.close();raise
    def _receive(self):
        deadline=time.monotonic()+self.timeout
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.selector.select(remaining):raise TimeoutError('retrieval worker response timed out')
            data=os.read(self.process.stdout.fileno(),65536)
            if not data:raise RuntimeError('retrieval worker exited; see retrieval_worker.stderr.log')
            self.buffer+=data
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)
    def query(self,image,top_k=10,exclude_ids=()):
        self.sequence+=1
        message=dict(id=self.sequence,image=str(Path(image).resolve()),top_k=top_k,exclude_ids=list(exclude_ids))
        try:
            self.process.stdin.write((json.dumps(message)+'\n').encode());self.process.stdin.flush()
            result=self._receive()
            if result.get('id')!=self.sequence:raise ValueError('retrieval worker response ID mismatch')
            if result.get('status')!='OK':raise RuntimeError(result.get('error','retrieval worker error'))
            return result
        except BaseException:
            self.close();raise
    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            for stream in (self.process.stdin,self.process.stdout):
                if stream is not None:
                    try:stream.close()
                    except (BrokenPipeError,OSError):pass
        if self.selector is not None:self.selector.close()
        if self.log is not None:self.log.close()


def shortlist_graph(graph,ranking,lookup):
    """Keep Top-K RGBs, grouped by their REAL graph node, without edges/history."""
    selected={};seen=set()
    for r in ranking:
        oid=r['image_id']
        if oid in seen or oid not in lookup:raise ValueError('invalid/duplicate ANN reference')
        ref=lookup[oid]
        if r['node_id']!=ref['node_id'] or not math.isfinite(r['similarity']):raise ValueError('invalid ANN node or score')
        seen.add(oid);selected.setdefault(ref['node_id'],[]).append(oid)
    if not selected:raise ValueError('empty retrieval shortlist')
    nodes={n['node_id']:n for n in graph['match_nodes']};result=[]
    for nid,refs in selected.items():
        if nid not in nodes or not set(refs)<=set(nodes[nid]['representative_observations']):
            raise ValueError('retrieval candidate does not belong to the current graph')
        node=copy.deepcopy(nodes[nid]);node['representative_observations']=refs;result.append(node)
    return dict(match_nodes=result,goal=graph['goal']),list(selected)


def ask_match_once(requests,payload,images,validator):
    """Use existing prompts/schema/backend and shared budget; no cache or repair."""
    kind='node_match';schema=requests.schemas[kind]
    records=[dict(id=k,**evidence_file(p)) for k,p in images]
    prompt=requests.common+'\n'+requests.prompts[kind]+'\n'+canonical(dict(payload=payload,image_ids=[k for k,_ in images],schema=schema))
    if len(prompt)>800_000:raise ValueError('retrieval matcher prompt too large')
    if requests.model_calls>=requests.max_calls:raise RuntimeError('graph experiment model-call budget exhausted')
    row=dict(kind=kind,matcher='retrieval_gpt_rerank',cache_hit=False,retry_count=0,model=requests.backend.model,
        images=records,prompt=prompt,prompt_hash=digest(prompt),schema_hash=digest(schema))
    start=time.perf_counter();requests.model_calls+=1
    try:
        response=requests.backend.invoke(prompt,[p for _,p in images],schema)
        requests.tool_metrics_available &= 'tool_calls' in response
        requests.tool_calls+=response.get('tool_calls',0);requests.visual_tool_calls+=response.get('visual_tool_calls',0)
        requests.completed_turns+=response.get('completed_turns',0)
        row.update(raw_response=response['text'],usage=response.get('usage'),recovered_errors=response.get('recovered_errors'))
        value=json.loads(response['text']);jsonschema.validate(value,schema);validator(value)
        row['structured_response']=value;return value
    except BaseException as exc:
        row['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        row['elapsed_s']=time.perf_counter()-start;requests.elapsed+=row['elapsed_s'];requests._log(row)


class RetrievalMatcher:
    def __init__(self,memory,graph_path,graph,trajectory,output,*,python=None,top_k=10,timeout=120.,worker_factory=RetrievalWorker):
        if top_k<1:raise ValueError('retrieval top_k must be positive')
        self.memory=validate_memory(memory,graph_path,graph,trajectory);self.top_k=top_k
        self.lookup={r['image_id']:r for r in self.memory['images']}
        self.worker=worker_factory(memory,output,python=python,timeout=timeout)
    def close(self):self.worker.close()
    def match(self,graph,trajectory,observation,requests,folder,metrics,*,pending=None,threshold=.8,exclude_ids=()):
        # Deliberately do not use pending/expected outcome to bias retrieval or GPT.
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
        row=dict(stage='retrieval_gpt_rerank',representative_images=True,top_k=self.top_k)
        for key in ('local_match_time','expanded_match_time','global_match_time','representative_image_fallback_count',
                    'representative_image_match_count','retrieval_match_time','retrieval_queries','gpt_rerank_time'):
            metrics.setdefault(key,0)
        try:
            retrieval=self.worker.query(observation.image,self.top_k,exclude_ids)
            save(folder/'retrieval.json',retrieval)
            if retrieval['query_encode_calls']!=1 or retrieval['reference_encode_calls_runtime']!=0:
                raise ValueError('retrieval worker violated query-only encoding contract')
            ranking=retrieval['ranking']
            if len(ranking)>self.top_k:raise ValueError('worker exceeded requested Top-K')
            subset,candidates=shortlist_graph(graph,ranking,self.lookup)
            images=[('current',observation.image)]+[(r['image_id'],checked_file(self.lookup[r['image_id']])) for r in ranking]
            row.update(candidate_nodes=candidates,image_ids=[k for k,_ in images],retrieval_timing=retrieval['timing'])
            metrics['retrieval_queries']+=1;metrics['representative_image_match_count']+=1
            for key,value in retrieval['timing'].items():metrics[key]=metrics.get(key,0.)+value
            start=time.perf_counter()
            try:
                value=ask_match_once(requests,signatures(subset,candidates,with_representatives=True),images,
                    lambda v:validate_match_response(v,dict(nodes=subset['match_nodes']),threshold))
            finally:
                row['gpt_rerank_s']=time.perf_counter()-start;metrics['gpt_rerank_time']+=row['gpt_rerank_s']
            row.update(response=value,decision='UNKNOWN' if value['status']=='NO_MATCH' else value['status'])
            return value  # Keep legacy NO_MATCH and real node IDs for replay dispatch gates.
        except BaseException as exc:
            row['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            row['elapsed_s']=time.perf_counter()-started;metrics['retrieval_match_time']+=row['elapsed_s']
            save(folder/'matching_attempts.json',dict(attempts=[row],time_unit='seconds',
                timing_scope='query-only retrieval worker plus one existing GPT request; GPT time includes attachment preparation'))


def main(argv=None):
    """Offline smoke test of the exact replay matcher; never connects hardware."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--memory',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--retrieval-python',type=Path);p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--gpt',action='store_true',help='one real GPT rerank; otherwise only exercise retrieval')
    p.add_argument('--exclude-self',action='store_true',help='exclude gallery files with the same RGB file hash in offline checks')
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--timeout-s',type=float,default=600.)
    args=p.parse_args(argv);args.output.mkdir(parents=True,exist_ok=False)
    from types import SimpleNamespace
    from .arm6_traj_replay import matching_graph,GARMENT_FRAME
    from .graph_policy import Requests
    from .graph_policy_schema import MATCH
    from .graph_policy_replay import PROMPTS
    graph=load_frozen(args.graph);trajectory=load_frozen(checked_file(graph['source_trajectory']))
    matcher=RetrievalMatcher(args.memory,args.graph,graph,trajectory,args.output,python=args.retrieval_python,top_k=args.top_k)
    try:
        excluded=[r['image_id'] for r in matcher.lookup.values() if r['sha256']==file_hash(args.image)] if args.exclude_self else []
        if args.gpt:
            from .discovery_model import CodexMultimodal
            requests=Requests(args.output/'model',CodexMultimodal(args.model,args.timeout_s),max_calls=1,no_cache=True,
                schemas={'node_match':MATCH},prompts={'node_match':PROMPTS['node_match']},
                common='You match cloth observations and ground an existing Arm6 demonstration. Images are attached in image_ids order. Return only schema-valid JSON. Do not generate robot code.\n'+GARMENT_FRAME)
            metrics={};value=matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=args.image),requests,args.output/'matching',metrics,exclude_ids=excluded)
            result=dict(match=value,model_metrics=requests.metrics(),matching_metrics=metrics,robot_calls=0)
        else:result=dict(retrieval=matcher.worker.query(args.image,args.top_k,excluded),robot_calls=0,model_calls=0)
        save(args.output/'result.json',result);print(json.dumps(result,ensure_ascii=False))
    finally:matcher.close()


if __name__=='__main__':main()
