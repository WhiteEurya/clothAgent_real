"""Render an evidence graph as a static PNG with reference thumbnails."""
import argparse
import html
import json
from pathlib import Path
import subprocess
import textwrap
from PIL import Image


def render(graph_path, output):
    output.mkdir(parents=True,exist_ok=True)
    g=json.loads(graph_path.read_text());t=json.loads(Path(g['source_trajectory']['path']).read_text())
    q=json.dumps
    lines=['digraph G {','rankdir=LR; bgcolor="#0c1525"; pad=0.4; nodesep=0.35; ranksep=0.8;',
           'graph [fontname="Noto Sans CJK SC", fontcolor="#edf3ff", fontsize=22, labelloc=t, label="Arm6 · 新旧示范合并图\\n灰：原动作  绿：新增辅助动作  紫：新增自由手动动作  橙：缺少后图\\n连线是观测到的示范；不代表每条边已可自动 replay"];',
           'node [shape=plain, fontname="Noto Sans CJK SC", fontcolor="#edf3ff"];',
           'edge [fontname="Noto Sans CJK SC", fontcolor="#b8cae4", fontsize=11, arrowsize=0.8];']
    for n in g['nodes']:
        nid=n['node_id'];refs=n['representative_observations'];color='#617897'
        if nid.startswith('x_'):color='#45b9a4'
        if nid in g.get('terminal_node_ids',[]):color='#68d391'
        if not refs:color='#edb55e'
        title=nid+(' · 目标' if nid in g.get('terminal_node_ids',[]) else '')
        label=f'<TABLE BORDER="2" COLOR="{color}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="7" BGCOLOR="#18283f"><TR><TD><FONT POINT-SIZE="18">{html.escape(title)}</FONT></TD></TR>'
        if refs:
            p=output/(nid+'.png');im=Image.open(t['observations'][refs[0]]['rgb']['path']).convert('RGB');im.thumbnail((220,124));im.save(p)
            label+=f'<TR><TD><IMG SRC="{html.escape(str(p.resolve()))}"/></TD></TR>'
        else:label+='<TR><TD>未观测结果</TD></TR>'
        text=n['task_context'];text=text[:60]+('…' if len(text)>60 else '')
        label+='<TR><TD><FONT POINT-SIZE="11">'+'<BR/>'.join(html.escape(v) for v in textwrap.wrap(text,18))+'</FONT></TD></TR>'
        label+=f'<TR><TD><FONT POINT-SIZE="10">参考照片 {len(refs)} 张</FONT></TD></TR></TABLE>'
        lines.append(f'{q(nid)} [label=<{label}>];')
    for e in g['edges']:
        new=e['seed_step_id'].startswith('new_');manual=e['seed_action']['exact_command']['type']=='ARM6_FREE_MANUAL_DEMONSTRATION'
        color='#bc9af5' if manual else '#45b9a4' if new else '#71859f'
        if not next(n for n in g['nodes'] if n['node_id']==e['target_node'])['representative_observations']:color='#edb55e'
        label=e['edge_id']+'\\n'+e['seed_step_id'].replace('new_assisted_','A/').replace('new_manual_','M/')
        lines.append(f'{q(e["source_node"])} -> {q(e["target_node"])} [label="{label}", color="{color}", fontcolor="{color}", penwidth={2 if new else 1}];')
    lines.append('}');dot=output/'overview.dot';dot.write_text('\n'.join(lines))
    subprocess.run(['dot','-Tpng',str(dot),'-o',str(output/'overview.png')],check=True)
    print(output/'overview.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();render(a.graph,a.output)
