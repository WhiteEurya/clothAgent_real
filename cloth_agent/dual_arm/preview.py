"""Standalone review artifact: both arm envelopes on the same world axes."""

from __future__ import annotations

import json
from pathlib import Path

from .planning import Motion
from .safety import padded_capsules


def write_preview(program, config, models, destination):
    samples = []
    for item in program:
        if not isinstance(item, Motion):
            continue
        stride = max(1, len(item.times) // 30)
        for i in sorted({*range(0, len(item.times), stride), len(item.times) - 1}):
            arms = {}
            envelopes = padded_capsules(
                config, models, {k: q[i] for k, q in item.joints.items()}
            )
            for k in config.arms:
                arms[k] = {
                    "tcp": item.poses[k][i][:3].tolist(),
                    "capsules": [
                        {"a": c.start.tolist(), "b": c.end.tolist(), "r": c.radius}
                        for c in envelopes[k]
                    ],
                }
            samples.append(
                {"phase": item.phase.name, "time": float(item.times[i]), "arms": arms}
            )
    data = json.dumps(
        {"samples": samples, "obstacles": config.obstacles}, allow_nan=False
    ).replace("<", "\\u003c")
    page = """<!doctype html><meta charset="utf-8"><title>Dual-arm trajectory preview</title>
<style>body{font:16px sans-serif;margin:24px;background:#151b24;color:#eee}canvas{background:#202a36;border-radius:8px}input{width:70%}label{margin:16px}</style>
<h1>Dual-arm trajectory preview</h1><p>World coordinates in mm. Left: blue; right: orange. Envelopes include configured geometry, tracking and stopping margins. Additional clearance is checked by the planner. Synthetic configurations have no physical error evidence. This is a planned trajectory, not execution evidence.</p>
<select id="view"><option value="1">Top: X/Y</option><option value="2">Front: X/Z</option></select>
<input id="slider" type="range" min="0" value="0"><p id="caption"></p><canvas id="scene" width="1100" height="720"></canvas>
<script>const data=DATA;const slider=document.getElementById('slider'),view=document.getElementById('view'),canvas=document.getElementById('scene'),ctx=canvas.getContext('2d');slider.max=data.samples.length-1;
function draw(){const s=data.samples[+slider.value],axis=+view.value;let points=[];for(const f of data.samples)for(const a of Object.values(f.arms))for(const c of a.capsules)points.push(c.a,c.b);let minX=Math.min(...points.map(p=>p[0]))-150,maxX=Math.max(...points.map(p=>p[0]))+150,minY=Math.min(...points.map(p=>p[axis]))-150,maxY=Math.max(...points.map(p=>p[axis]))+150;let scale=Math.min(1000/(maxX-minX),620/(maxY-minY));const project=p=>[50+(p[0]-minX)*scale,670-(p[axis]-minY)*scale];ctx.clearRect(0,0,1100,720);ctx.strokeStyle='#45566b';ctx.lineWidth=1;for(let x=Math.ceil(minX/100)*100;x<maxX;x+=100){let p=project([x,0,0]);ctx.beginPath();ctx.moveTo(p[0],30);ctx.lineTo(p[0],680);ctx.stroke();ctx.fillStyle='#b8c8dd';ctx.fillText(x,p[0],704)}for(const b of data.obstacles){let a=project(b.min_mm),z=project(b.max_mm);ctx.fillStyle='#8886';ctx.fillRect(a[0],z[1],z[0]-a[0],a[1]-z[1])}for(const [k,a] of Object.entries(s.arms)){let color=k==='left'?'#54b9ff':'#ffa84d';for(const c of a.capsules){let p=project(c.a),q=project(c.b);ctx.strokeStyle=color+'80';ctx.lineWidth=2*c.r*scale;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke()}let p=project(a.tcp);ctx.fillStyle=color;ctx.beginPath();ctx.arc(...p,5,0,2*Math.PI);ctx.fill();ctx.fillText(k,p[0]+10,p[1]-10)}document.getElementById('caption').textContent=s.phase+' | phase time '+s.time.toFixed(2)+' s | frame '+slider.value+'/'+slider.max}slider.oninput=draw;view.onchange=draw;draw();</script>"""
    Path(destination).write_text(page.replace("DATA", data))
