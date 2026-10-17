"""Synchronized RGB/IR multiview evidence. Camera only; no calibration writes."""
import argparse
from datetime import datetime,timezone
import fcntl
from pathlib import Path
import time
import tkinter as tk
from PIL import Image,ImageTk
from cloth_agent.experiments.registration_field import NativeCamera,ROOT,save

GUIDES=[
 '同一张板：中央、桌面附近，图案斜约30°，整板可见。每个姿态停稳2秒。',
 '在桌面附近移动到画面左侧、再到右侧；不要碰相机或机械臂。',
 '板向相机靠近约10–15 cm；保持整板可见，停稳后左右移动。',
 '在较近位置，板左右倾斜约20–30°，分别停稳2秒。',
 '在较近位置，板前后倾斜约20–30°，分别停稳2秒。',
 '回到桌面附近，左右及前后倾斜约20–30°，分别停稳2秒。',
 '保持倾斜，覆盖画面上部、下部；整板可见，手不要遮住tag。',
 '补充不同距离与倾角：每次移动后停稳2秒，不要只平移同一姿态。',
 '回到中央，较远/较近各停稳一次；等待采集完成。',
]

def guide(elapsed,duration):
    return GUIDES[min(len(GUIDES)-1,int(max(0,elapsed)/duration*len(GUIDES)))]

def run(args):
    out=args.output/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='STARTING',robot_connected=False,calibration_written=False,
        data_use='RGB/IR geometric diagnostic only; emitter off, no usable depth assumed',
        board_outer_black_span_mm=args.board_span_mm,duration_s=args.duration,interval_s=args.interval,frames=[])
    save(out/'session.json',manifest)
    window=None;camera=None;lease=None;started=None;next_frame=0.;stopping=False
    try:
        window=tk.Tk();window.title('Camera B 多视角诊断 · 仅相机 · 不写标定')
        status=tk.Label(window,text='初始化相机……',font=('Noto Sans CJK SC',14),wraplength=1080);status.pack(pady=8)
        note=tk.Label(window,text='只放同一张142 mm板。准备好后按Enter开始90秒；移动后停稳2秒。Esc退出并恢复设置。',font=('Noto Sans CJK SC',11));note.pack()
        picture=tk.Label(window);picture.pack()
        def begin(event=None):
            nonlocal started,next_frame
            if started is None:
                started=time.monotonic();next_frame=started;manifest['status']='CAPTURING';save(out/'session.json',manifest)
        def stop(event=None):
            nonlocal stopping
            stopping=True
        tk.Button(window,text='准备好，开始（Enter）',command=begin).pack(side='left',padx=20,pady=8)
        tk.Button(window,text='停止并恢复（Esc）',command=stop).pack(side='right',padx=20,pady=8)
        window.bind('<Return>',begin);window.bind('<Escape>',stop);window.protocol('WM_DELETE_WINDOW',stop)
        lease=open('/tmp/clothagent_codex_dual_arm.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        camera=NativeCamera(args.config);meta=camera.start();save(out/'camera.json',meta)
        camera.emitter(0)
        for _ in range(30):camera.read()
        manifest['status']='WAITING_FOR_OPERATOR';save(out/'session.json',manifest)
        while not stopping:
            window.update_idletasks();window.update()
            f=camera.read();now=time.monotonic()
            elapsed=0 if started is None else now-started
            if started is not None and elapsed>=args.duration:
                manifest['status']='COMPLETE';break
            text='等待Enter开始。只保留一张板，整板可见，不触碰机械臂。' if started is None else guide(elapsed,args.duration)
            status.config(text=f'{elapsed:.0f}/{args.duration:g}秒 · 已存{len(manifest["frames"])}组\n{text}')
            image=Image.fromarray(f['rgb']);image.thumbnail((1080,608));photo=ImageTk.PhotoImage(image);picture.config(image=photo);picture.image=photo
            if started is not None and now>=next_frame:
                idx=len(manifest['frames']);folder=out/f'frame_{idx:04d}';folder.mkdir()
                for name in ['rgb','ir1','ir2']:Image.fromarray(f[name]).save(folder/f'{name}.png')
                entry=dict(index=idx,elapsed_s=elapsed,timestamps=f['timestamps'],instruction=text)
                save(folder/'frame.json',entry);manifest['frames'].append(entry);save(out/'session.json',manifest)
                next_frame=now+args.interval
        if manifest['status']!='COMPLETE':manifest['status']='INTERRUPTED'
    except (Exception,KeyboardInterrupt) as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    finally:
        if camera is not None:
            try:save(out/'restoration.json',camera.close())
            except Exception as exc:manifest.update(status='RESTORATION_FAILED',error=str(exc))
        if lease is not None:lease.close()
        if window is not None:
            try:window.destroy()
            except tk.TclError:pass
        save(out/'session.json',manifest)
        print(f'{manifest["status"]}: {out}',flush=True)
        if manifest.get('error'):print(manifest['error'],flush=True)
    return 0 if manifest['status']=='COMPLETE' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'config/dual_arm.local.json')
    p.add_argument('--output',type=Path,default=ROOT/'runs/camb_multiview_capture')
    p.add_argument('--duration',type=float,default=90);p.add_argument('--interval',type=float,default=.5)
    p.add_argument('--board-span-mm',type=float,default=142.)
    a=p.parse_args()
    import math
    if not all(math.isfinite(v) and v>0 for v in [a.duration,a.interval,a.board_span_mm]) or a.interval>a.duration:p.error('Require positive finite duration/interval/span and interval <= duration')
    return run(a)

if __name__=='__main__':raise SystemExit(main())
