"""Independent local dense proposal probe.

V2: prepare-v2 / discover / review / freeze-groups / localize / summarize.
The former exhaustive-ROI annotation commands remain available only for old data.

Legacy annotation-only commands below do not compute features.
V2 runs frozen dense features but never GPT, cameras, robots, or graph runtime.
prepare --output DIR
annotate --experiment DIR
validate --experiment DIR
freeze --experiment DIR
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

SOURCE=Path('reports/arm6_local_action_20261020/memory/library.json')
DEMO_SOURCE=Path('reports/local_experience_transfer_probe_20261020/dataset')
CHECKPOINT=Path('/mnt/newssd/sja/clothAgent_real/model_assets/local_node_benchmark/dinov2-small')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def pixel_hash(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        return hashlib.sha256(f'{im.width}x{im.height}:RGB:'.encode()+im.tobytes()).hexdigest()


def resize_mapping(size):
    w,h=size;scale=896/max(w,h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
    pw,ph=math.ceil(rw/14)*14,math.ceil(rh/14)*14
    return dict(original_size=[w,h],resized_size=[rw,rh],padded_size=[pw,ph],
                scale_xy=[rw/w,rh/h],padding_ltrb=[0,0,pw-rw,ph-rh],grid_wh=[pw//14,ph//14],
                patch_center_to_original='x=((col+0.5)*14)/scale_x; y=((row+0.5)*14)/scale_y')


def protocol(checkpoint):
    cfg=read(checkpoint/'config.json');prep=read(checkpoint/'preprocessor_config.json')
    if cfg['model_type']!='dinov2' or cfg['patch_size']!=14 or cfg['hidden_size']!=384:raise ValueError('expected frozen DINOv2-small')
    return dict(model_path=str(checkpoint.resolve()),checkpoint_files={p.name:sha(p) for p in sorted(checkpoint.iterdir()) if p.is_file()},
        training=False,model_loading_performed=False,feature_computation_performed=False,
        preprocess=dict(long_edge_px=896,interpolation='PIL bicubic',aspect='preserve; round resized dimensions',
            padding='right/bottom to multiple of 14, zero after channel normalization',
            rescale_factor=1/255,image_mean=prep['image_mean'],image_std=prep['image_std'],center_crop=False),
        tokens='last hidden patch tokens only; exclude CLS; L2 normalize each patch vector',
        A='descriptor of patch containing Pa_demo; cosine with all query patch descriptors',
        B='3x3 mean of L2-normalized patch tokens around anchor patch, then L2 normalize',
        C='L2-normalized 3x3 core and L2-normalized 9x9 context, concatenate with equal weight, L2 normalize again',
        windows='same pooling for demo/query; truncate at valid patch-center boundary and divide by actual valid count; no cloth mask',
        invalid_centers='ignore patch centers outside resized image content (padding only)',
        demo_anchor='floor(Pa_demo * scale_xy / 14), row/column respectively',
        nms=dict(radius_original_px=56,metric='Euclidean',suppress_if='distance <= radius',top_k=[1,5,10],
                 ordering='descending cosine; ties row-major',score_threshold=None),
        evaluation=dict(positive_hit='candidate center inside or on any frozen admissible polygon/rectangle',
            positive_distance='minimum original-pixel Euclidean distance from any Top-K center to union of admissible regions; zero inside',
            subgroup='same-session and cross-session separately',
            negative='no similarity threshold; retain Top-1 cosine, raw peak-minus-next-patch and peak-minus-second-NMS-anchor margins, and positions'),
        tuning=False,forbidden=['GPT','depth','edge detector','cloth mask','rotation normalization','training','CLS retrieval'])


def candidates(library,excluded_hashes):
    by_session=defaultdict(list);seen=set(excluded_hashes)
    for eid,a in sorted(library['actions'].items()):
        if not a['session'].startswith('20261019'):continue
        for phase in ('before','after'):
            ref=a[phase];p=Path(ref['path'])
            if sha(p)!=ref['sha256']:raise ValueError('source changed: '+str(p))
            ph=pixel_hash(p)
            if ph in seen:continue
            seen.add(ph)
            by_session[a['session']].append(dict(source=str(p.resolve()),source_sha256=sha(p),pixel_sha256=ph,
                                                 session=a['session'],trajectory_evidence_id=eid,phase=phase))
    return by_session


def select_queries(groups,demo_session):
    same=groups[demo_session]
    if len(same)<4:raise ValueError('not enough unique same-session observations')
    chosen=[same[round(i*(len(same)-1)/3)] for i in range(4)]
    other={s:v for s,v in sorted(groups.items()) if s!=demo_session}
    # Session coverage first, then terminal observations from largest sessions.
    cross=[items[len(items)//2] for items in other.values()]
    if len(cross)>12:cross=[cross[round(i*(len(cross)-1)/11)] for i in range(12)]
    selected={v['pixel_sha256'] for v in chosen+cross}
    for session,items in sorted(other.items(),key=lambda pair:(-len(pair[1]),pair[0])):
        for item in reversed(items):
            if len(cross)>=12:break
            if item['pixel_sha256'] not in selected:
                cross.append(item);selected.add(item['pixel_sha256']);break
    if len(cross)!=12:raise ValueError('not enough cross-session observations')
    return chosen+cross


def prepare(output,library=SOURCE,demo_source=DEMO_SOURCE,checkpoint=CHECKPOINT):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    lib=read(library);prior=read(demo_source/'manifest.json')
    demo=lib['actions']['e095'];session=demo['session']
    source_demo=demo_source/'public/demo_before.png'
    if pixel_hash(source_demo)!=pixel_hash(demo['before']['path']):raise ValueError('demo differs from previous probe')
    excluded={pixel_hash(source_demo),pixel_hash(demo_source/'public/demo_after.png')}
    selected=select_queries(candidates(lib,excluded),session)
    files={}
    for name in ('demo_before.png','demo_action_overlay.png','demo_after.png'):
        destination=root/'images'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(demo_source/'public'/name) as im:im.convert('RGB').save(destination)
        files[str(destination.relative_to(root))]=sha(destination)
    records=[]
    for i,item in enumerate(selected,1):
        qid=f'q{i:02d}';relative=f'images/{qid}.png';p=root/relative
        with Image.open(item['source']) as im:
            im=im.convert('RGB');size=list(im.size);im.save(p)
        files[relative]=sha(p)
        records.append(dict(id=qid,image=relative,size=size,subset='same-session' if item['session']==session else 'cross-session',
                            resize_mapping=resize_mapping(size),**item))
    pp=protocol(Path(checkpoint));save(root/'protocol.json',pp)
    with Image.open(source_demo) as im:demo_size=list(im.size)
    manifest=dict(experiment='local_dense_proposal_probe',stage='AWAITING_HUMAN_ANNOTATION',
        demo=dict(edge_id='e095',session=session,pa_demo=prior['demo_pa'],size=demo_size,resize_mapping=resize_mapping(demo_size),
                  image='images/demo_before.png',historical_overlay='images/demo_action_overlay.png',after='images/demo_after.png'),
        queries=records,files=files,protocol_sha256=sha(root/'protocol.json'),
        selection='No image model or class labels used. Four evenly spaced unique observations within demo session; one midpoint observation per other session, then terminal observations from largest sessions to reach twelve cross-session queries.',
        source_library=dict(path=str(Path(library).resolve()),sha256=sha(library)),
        no_existing_proxy_labels_imported=True,model_calls=0,feature_computations=0,robot_calls=0,created_unix=time.time())
    save(root/'manifest.json',manifest)
    save(root/'annotations.draft.json',dict(annotation_origin='HUMAN_REQUIRED',annotator='',
        manifest_sha256=sha(root/'manifest.json'),queries={q['id']:dict(label='UNREVIEWED',regions=[],reviewed_all_regions=False) for q in records}))
    # Overview contains no inferred or suggested ground-truth labels.
    sheet=Image.new('RGB',(1280,4*220),'#111a29');d=ImageDraw.Draw(sheet)
    f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for i,q in enumerate(records):
        x=(i%4)*320;y=(i//4)*220
        with Image.open(root/q['image']) as im:im=ImageOps.contain(im.convert('RGB'),(318,179))
        sheet.paste(im,(x,y+40));d.text((x+5,y+5),q['id']+' / '+q['subset'],font=f,fill='white')
        d.text((x+5,y+23),q['session'],font=f,fill='#b1c2da')
    sheet.save(root/'query_overview.jpg')
    (root/'README.txt').write_text('当前阶段：等待人工标注。没有计算任何 DINO 特征。\n'
        '运行：python -m cloth_agent.experiments.local_dense_proposal_probe annotate --experiment '+str(root)+'\n'
        '每张 positive 请标出所有允许复用本 experience 的区域，可添加多个矩形/多边形。\n'
        'negative 选择 NOT_APPLICABLE；不要只标一个错误 candidate 就把整图判为 negative。\n'
        '每张完成后勾选已检查整张图。保存草稿后可关闭并恢复。\n'
        '全部完成后点击冻结；也可运行同模块 freeze --experiment DIR。\n'
        '冻结只封存人工标注，不启动特征计算。没有实现或运行 feature benchmark。\n')
    return dict(status='AWAITING_HUMAN_ANNOTATION',queries=16,same_session=4,cross_session=12,feature_computations=0)


def verify_inputs(root):
    m=read(root/'manifest.json')
    if sha(root/'protocol.json')!=m['protocol_sha256']:raise ValueError('protocol changed')
    for name,expected in m['files'].items():
        if sha(root/name)!=expected:raise ValueError('image changed: '+name)
    return m


def annotation_errors(manifest,annotations):
    errors=[]
    if annotations.get('annotation_origin')!='HUMAN_REQUIRED' or not annotations.get('annotator','').strip():errors.append('请填写人工标注者姓名/代号')
    if set(annotations.get('queries',{}))!={q['id'] for q in manifest['queries']}:return errors+['query ID 集合不一致']
    for q in manifest['queries']:
        a=annotations['queries'][q['id']];prefix=q['id']+': '
        if a.get('label') not in ('POSITIVE','NOT_APPLICABLE'):errors.append(prefix+'尚未选择标签');continue
        if a.get('reviewed_all_regions') is not True:errors.append(prefix+'尚未确认检查整张图')
        if a['label']=='NOT_APPLICABLE' and a.get('regions'):errors.append(prefix+'negative 不能含允许区域')
        if a['label']=='POSITIVE' and not a.get('regions'):errors.append(prefix+'positive 至少需要一个区域')
        for region in a.get('regions',[]):
            points=region.get('polygon_px',[])
            if region.get('type') not in ('rectangle','polygon') or len(points)<3:errors.append(prefix+'无效区域');continue
            try:
                valid=all(len(p)==2 and all(math.isfinite(v) for v in p) and 0<=p[0]<q['size'][0] and 0<=p[1]<q['size'][1] for p in points)
                area=abs(sum(p[0]*n[1]-n[0]*p[1] for p,n in zip(points,points[1:]+points[:1])))/2
                if not valid or area<4:errors.append(prefix+'区域越界或退化')
            except (TypeError,ValueError):errors.append(prefix+'区域坐标格式错误')
    return errors


def freeze(root):
    root=Path(root);m=verify_inputs(root);a=read(root/'annotations.draft.json')
    if a['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('annotation manifest mismatch')
    errors=annotation_errors(m,a)
    if errors:raise ValueError('\n'.join(errors))
    target=root/'annotations.frozen.json'
    if target.exists():raise ValueError('已冻结；不会覆盖已有标注')
    save(target,dict(**a,frozen_unix=time.time(),human_attestation='Annotator explicitly reviewed all images and admissible regions before any features were computed.'))
    save(root/'annotation_freeze.json',dict(manifest_sha256=sha(root/'manifest.json'),protocol_sha256=sha(root/'protocol.json'),
        annotations_sha256=sha(target),status='ANNOTATED_NOT_BENCHMARKED',feature_computations=0))
    return target


def canvas_to_original(x,y,scale,offset,size):
    px=(x-offset[0])/scale;py=(y-offset[1])/scale
    return [max(0,min(size[0]-1,px)),max(0,min(size[1]-1,py))]


class Annotator:
    def __init__(self,root):
        import tkinter as tk
        from tkinter import ttk,messagebox
        from PIL import ImageTk
        self.tk=tk;self.ttk=ttk;self.messagebox=messagebox;self.ImageTk=ImageTk
        self.root=Path(root);self.m=verify_inputs(self.root)
        if (self.root/'annotations.frozen.json').exists():raise ValueError('已冻结，不允许在原实验上修改标注')
        self.a=read(self.root/'annotations.draft.json');self.index=0;self.vertices=[];self.drag=None
        self.win=tk.Tk();self.win.title('Local dense proposal — 人工标注（未计算特征）');self.win.geometry('1450x1000')
        self.name=tk.StringVar(value=self.a['annotator']);self.label=tk.StringVar();self.reviewed=tk.BooleanVar();self.mode=tk.StringVar(value='rectangle')
        top=ttk.Frame(self.win);top.pack(fill='x');self.demo_photos=[]
        for name,title in [('demo_before.png','Demo BEFORE'),('demo_action_overlay.png','历史 Pa / movement'),('demo_after.png','Demo AFTER')]:
            frame=ttk.Frame(top);frame.pack(side='left',padx=8)
            with Image.open(self.root/'images'/name) as im:im=ImageOps.contain(im.convert('RGB'),(380,214))
            photo=ImageTk.PhotoImage(im,master=self.win);self.demo_photos.append(photo)
            ttk.Label(frame,text=title).pack();ttk.Label(frame,image=photo).pack()
        bar=ttk.Frame(self.win);bar.pack(fill='x',pady=4)
        ttk.Label(bar,text='标注者：').pack(side='left');ttk.Entry(bar,textvariable=self.name,width=16).pack(side='left')
        ttk.Label(bar,text=' 标签：').pack(side='left')
        for value in ('UNREVIEWED','POSITIVE','NOT_APPLICABLE'):
            ttk.Radiobutton(bar,text=value,variable=self.label,value=value,command=self.label_changed).pack(side='left')
        ttk.Checkbutton(bar,text='已检查整张图：区域已标全 / 确认不适用',variable=self.reviewed).pack(side='left',padx=10)
        controls=ttk.Frame(self.win);controls.pack(fill='x')
        for value,title in [('rectangle','拖动矩形'),('polygon','点击多边形')]:
            ttk.Radiobutton(controls,text=title,variable=self.mode,value=value,command=self.cancel_shape).pack(side='left')
        for title,fn in [('闭合多边形 (Enter)',self.close_polygon),('撤销最后区域',self.undo),('清空本图',self.clear),('上一张',lambda:self.navigate(-1)),('保存草稿',self.store),('下一张',lambda:self.navigate(1)),('检查并冻结',self.finish)]:
            ttk.Button(controls,text=title,command=fn).pack(side='left',padx=3)
        ttk.Label(self.win,text='标出所有允许复用 demo 操作的局部区域，可添加多个区域；不要仅标“能抓住”的位置。Esc 取消正在绘制的形状。').pack()
        self.info=ttk.Label(self.win);self.info.pack()
        self.canvas=tk.Canvas(self.win,bg='#111a29',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.redraw());self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.motion);self.canvas.bind('<ButtonRelease-1>',self.release)
        self.win.bind('<Return>',lambda e:self.close_polygon());self.win.bind('<Escape>',lambda e:self.cancel_shape())
        self.win.protocol('WM_DELETE_WINDOW',self.quit);self.load()

    def current(self):return self.m['queries'][self.index]
    def annotation(self):return self.a['queries'][self.current()['id']]
    def load(self):
        a=self.annotation();self.label.set(a['label']);self.reviewed.set(a['reviewed_all_regions']);self.vertices=[];self.drag=None
        with Image.open(self.root/self.current()['image']) as im:self.original=im.convert('RGB')
        self.redraw()
    def store(self):
        a=self.annotation();a['label']=self.label.get();a['reviewed_all_regions']=self.reviewed.get()
        self.a['annotator']=self.name.get().strip();self.a['last_saved_unix']=time.time();save(self.root/'annotations.draft.json',self.a)
    def navigate(self,delta):
        self.store();self.index=max(0,min(len(self.m['queries'])-1,self.index+delta));self.load()
    def label_changed(self):
        self.reviewed.set(False);self.redraw()
    def cancel_shape(self):self.vertices=[];self.drag=None;self.redraw()
    def add(self,points,kind):
        self.annotation()['regions'].append(dict(type=kind,polygon_px=points));self.label.set('POSITIVE');self.reviewed.set(False);self.store();self.cancel_shape()
    def close_polygon(self):
        if len(self.vertices)>=3:self.add(self.vertices[:],'polygon')
    def undo(self):
        if self.vertices:self.vertices.pop()
        elif self.annotation()['regions']:self.annotation()['regions'].pop();self.reviewed.set(False);self.store()
        self.redraw()
    def clear(self):
        if self.messagebox.askyesno('清空','清空本图全部区域？',parent=self.win):self.annotation()['regions']=[];self.reviewed.set(False);self.store();self.cancel_shape()
    def point(self,e):return canvas_to_original(e.x,e.y,self.scale,self.offset,self.current()['size'])
    def press(self,e):
        if self.mode.get()=='rectangle':self.drag=self.point(e)
    def motion(self,e):
        if self.drag is not None:self.redraw();x,y=self.to_canvas(self.drag);u,v=self.to_canvas(self.point(e));self.canvas.create_rectangle(x,y,u,v,outline='#ffda65',width=2)
    def release(self,e):
        if self.mode.get()=='polygon':self.vertices.append(self.point(e));self.redraw()
        elif self.drag is not None:
            x,y=self.drag;u,v=self.point(e)
            if abs(u-x)>=2 and abs(v-y)>=2:self.add([[min(x,u),min(y,v)],[max(x,u),min(y,v)],[max(x,u),max(y,v)],[min(x,u),max(y,v)]],'rectangle')
            else:self.cancel_shape()
    def to_canvas(self,p):return [p[0]*self.scale+self.offset[0],p[1]*self.scale+self.offset[1]]
    def redraw(self):
        if not hasattr(self,'original'):return
        w=max(1,self.canvas.winfo_width());h=max(1,self.canvas.winfo_height());ow,oh=self.original.size
        self.scale=min(w/ow,h/oh);rw,rh=max(1,round(ow*self.scale)),max(1,round(oh*self.scale));self.offset=((w-rw)/2,(h-rh)/2)
        self.photo=self.ImageTk.PhotoImage(self.original.resize((rw,rh)),master=self.win)
        self.canvas.delete('all');self.canvas.create_image(*self.offset,anchor='nw',image=self.photo)
        for i,r in enumerate(self.annotation()['regions'],1):
            coords=[v for p in r['polygon_px'] for v in self.to_canvas(p)]
            self.canvas.create_polygon(*coords,fill='',outline='#44edb5',width=3)
            self.canvas.create_text(*self.to_canvas(r['polygon_px'][0]),text=str(i),fill='yellow',anchor='nw')
        for p in self.vertices:
            x,y=self.to_canvas(p);self.canvas.create_oval(x-3,y-3,x+3,y+3,fill='yellow')
        if len(self.vertices)>1:self.canvas.create_line(*[v for p in self.vertices for v in self.to_canvas(p)],fill='yellow',width=2)
        q=self.current();self.info.config(text=f'{self.index+1}/{len(self.m["queries"])}  {q["id"]}  {q["subset"]}  {q["session"]}  区域数={len(self.annotation()["regions"])}')
    def finish(self):
        self.store()
        errors=annotation_errors(self.m,self.a)
        if errors:self.messagebox.showerror('尚不能冻结','\n'.join(errors[:25]),parent=self.win);return
        if not self.messagebox.askyesno('冻结人工标注','确认所有图已人工检查，所有允许区域已标全？冻结后不可覆盖；不会开始计算特征。',parent=self.win):return
        path=freeze(self.root);self.messagebox.showinfo('已冻结',str(path)+'\n未启动特征计算。',parent=self.win);self.win.destroy()
    def quit(self):self.store();self.win.destroy()


def main(argv=None):
    import sys
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('prepare-v2','discover','review','freeze-groups','localize','summarize'):
        from .local_dense_proposal_v2 import main as v2_main
        return v2_main(argv)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',type=Path,required=True)
    for name in ('annotate','validate','freeze'):
        a=sub.add_parser(name);a.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='prepare':print(json.dumps(prepare(a.output),ensure_ascii=False,indent=2))
    elif a.command=='annotate':
        ui=Annotator(a.experiment);ui.win.mainloop()
    elif a.command=='freeze':print(freeze(a.experiment))
    else:
        m=verify_inputs(a.experiment);errors=annotation_errors(m,read(a.experiment/'annotations.draft.json'))
        print(json.dumps(dict(ready_to_freeze=not errors,errors=errors,feature_computations=0),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
