"""Read-only replay telemetry. No robot interfaces and no external browser assets."""
import functools
import hashlib
import json
import logging
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time


def projection(graph):
    if isinstance(graph.get('nodes'), dict):
        nodes=[dict(id=k,label=v['node'].get('task_context',k),status=v.get('status','EXISTING'),
                    references=v['node'].get('representative_observations',[])) for k,v in graph['nodes'].items()]
        actions=[dict(id=k,source=v['source_node'],targets=list(v.get('outcomes',{})),
                      target=v.get('nominal_target'),preferred=v.get('preferred',False),
                      outcomes=v.get('outcomes',{})) for k,v in graph['actions'].items()]
    else:
        nodes=[dict(id=n['node_id'],label=n.get('task_context',''),status='EXISTING',
                    references=n.get('representative_observations',[])) for n in graph['nodes']]
        actions=[dict(id=e['edge_id'],source=e['source_node'],targets=[e['target_node']],target=e['target_node'],
                      preferred=e.get('replayable',False),outcome=e.get('seed_outcome','')) for e in graph['edges']]
    return dict(nodes=nodes,actions=actions)


class LiveGraph:
    def __init__(self, output, port=8090):
        self.folder=Path(output)/'live_graph';self.folder.mkdir(parents=True,exist_ok=True)
        self.state=dict(current=None,action=None,target=None,phase='STARTING',new_nodes=[],events=[],recoveries=[])
        self.known=None;self.server=None
        self.observations={};self.node_observations={};self.image_cache={};self.assets=set()
        try:
            owner=self
            class Handler(SimpleHTTPRequestHandler):
                def log_message(self,*args): pass
                def do_GET(self):
                    if self.path.split('?')[0] not in {'/','/index.html','/state.json'}|owner.assets:
                        self.send_error(404);return
                    super().do_GET()
                def do_HEAD(self):
                    if self.path.split('?')[0] not in {'/','/index.html','/state.json'}|owner.assets:
                        self.send_error(404);return
                    super().do_HEAD()
                def end_headers(self):
                    self.send_header('Cache-Control','no-store');super().end_headers()
            self.server=ThreadingHTTPServer(('127.0.0.1',port),functools.partial(Handler,directory=str(self.folder)))
            Thread(target=self.server.serve_forever,daemon=True).start()
            self.url=f'http://127.0.0.1:{self.server.server_port}'
            print('LIVE GRAPH: '+self.url+' (只读)',flush=True)
        except OSError as exc:
            self.url=None
            logging.warning('Live graph HTTP unavailable; snapshots still saved: %s',exc)

    def update(self, graph, **fields):
        # Telemetry must never reject or dispatch a robot action.
        try:
            self.observations.update(fields.pop('observations',{}))
            self.observations.update(graph.get('observations',{}))
            observation=fields.pop('observation',None)
            if observation is not None and fields.get('current') is not None:
                self.node_observations[fields['current']]=observation
            data=projection(graph)
            data['nodes']+=[dict(n) for n in fields.get('extra_nodes',self.state.get('extra_nodes',[]))]
            data['actions']+=fields.get('extra_actions',self.state.get('extra_actions',[]))
            for node in data['nodes']:
                images=[]
                if node['id'] in self.node_observations:
                    images.append(self._image(self.node_observations[node['id']],'最近匹配观测'))
                for ref in node.pop('references',[]):
                    images.append(self._image(self.observations.get(ref,{}).get('rgb'),f'节点参考 · {ref}'))
                if node.get('representative_observation'):
                    images.append(self._image(node['representative_observation'],'节点参考'))
                node['images']=images
            if 'recoveries' in fields:
                fields['recoveries']=[dict(row,images=[
                    self._image(row.get('before'),'恢复前'),
                    self._image(row.get('after'),'恢复后')]) for row in fields['recoveries']]
                for row in fields['recoveries']:
                    row.pop('before',None);row.pop('after',None)
            ids={n['id'] for n in data['nodes']}
            added=sorted(ids-self.known) if self.known is not None else []
            self.known=ids
            if added:
                self.state['new_nodes']+=added
                self.state['events'].append(dict(time=time.time(),new_nodes=added))
            self.state.update(data,**fields,updated=time.time())
            raw=json.dumps(self.state,ensure_ascii=False)
            self._write('state.json',raw)
            self._write('index.html',PAGE.replace('__SNAPSHOT__',raw.replace('<','\\u003c')))
        except Exception:
            logging.exception('Live graph update failed; replay is unaffected')

    def _image(self, evidence, label):
        """Export only explicitly referenced images; the HTTP endpoint accepts no paths."""
        image=dict(label=label,src=None)
        if isinstance(evidence,dict):evidence=evidence.get('rgb',evidence)
        if isinstance(evidence,dict):evidence=evidence.get('path')
        if not evidence:return image
        try:
            path=Path(evidence).resolve();stat=path.stat()
            key=(str(path),stat.st_mtime_ns,stat.st_size)
            if key not in self.image_cache:
                from PIL import Image, ImageOps
                name=hashlib.sha256(repr(key).encode()).hexdigest()+'.jpg'
                target=self.folder/'images'/name;target.parent.mkdir(exist_ok=True)
                with Image.open(path) as original:
                    preview=ImageOps.exif_transpose(original).convert('RGB')
                    preview.thumbnail((1600,1600))
                    temp=target.with_suffix('.tmp');preview.save(temp,format='JPEG',quality=90);temp.replace(target)
                self.image_cache[key]='images/'+name
                self.assets.add('/images/'+name)
            image['src']=self.image_cache[key]
        except (OSError,ValueError,TypeError):
            pass  # Missing evidence stays explicit in the viewer, without blocking telemetry.
        return image

    def _write(self,name,content):
        p=self.folder/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(content);t.replace(p)

    def close(self):
        if self.server:
            self.server.shutdown();self.server.server_close()


PAGE=Path(__file__).with_suffix('.html').read_text(encoding='utf-8')


class NullGraph:
    def update(self,*args,**kwargs): pass
    def close(self): pass


def create_live_graph(output,port):
    if port is None:return NullGraph()
    try:return LiveGraph(output,port)
    except Exception:
        logging.exception('Live graph unavailable; replay is unaffected')
        return NullGraph()
