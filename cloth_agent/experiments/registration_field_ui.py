"""Chinese Tk placement guide; UI never owns camera/robot controls."""
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import webbrowser
from PIL import Image,ImageDraw,ImageFont,ImageTk
from cloth_agent.experiments.registration_field import PLACEMENTS,TARGETS

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def overlay(rgb,placement,center=None,tags=None):
    image=Image.fromarray(rgb).convert('RGB');draw=ImageDraw.Draw(image,'RGBA')
    w,h=image.size;font=ImageFont.truetype(FONT,20)
    labels={'L-M':'左侧 · 中排','C-M':'中央 · 中排','R-M':'右侧 · 中排'}
    for label,(tx,ty) in TARGETS.items():
        x,y=tx*w,ty*h;active=label==placement
        color=(40,255,100,255) if active else (190,190,190,170)
        # Bracket is a CENTER acceptance region, not the required board footprint.
        box=(x-.065*w,y-.10*h,x+.065*w,y+.10*h)
        draw.rectangle(box,fill=(30,180,80,35) if active else (0,0,0,0),outline=color,width=3 if active else 1)
        draw.line((x-12,y,x+12,y),fill=color,width=2);draw.line((x,y-12,x,y+12),fill=color,width=2)
        draw.text((x-70,y-.10*h-30),label+' '+labels[label],font=font,fill=color)
    draw.rectangle((0,0,w,65),fill=(0,0,0,185))
    draw.text((12,4),'把整张 4×4 图案的中心放到绿色框内；绿色框不是板的边界',font=font,fill='white')
    draw.text((12,32),'板平放，图案边相对画面水平线斜约30°；三处保持同一朝向，勿触碰机械臂',font=font,fill='white')
    draw.rectangle((w-290,h-125,w,h),fill=(0,0,0,185))
    draw.text((w-280,h-120),'摆放朝向示意（非尺寸）',font=font,fill='white')
    draw.line((w-250,h-40,w-80,h-40),fill=(180,180,180,255),width=2)
    draw.polygon([(w-210,h-90),(w-123,h-40),(w-148,h+3),(w-235,h-47)],outline=(255,230,0,255),width=3)
    if center is not None:
        x,y=center;draw.ellipse((x-9,y-9,x+9,y+9),outline=(255,230,0,255),width=3)
        draw.text((x+12,y+8),f'板中心 · 检出 {tags}/16 tags',font=font,fill=(255,230,0,255))
    return image

class Interface:
    def __init__(self,root):
        self.root=root;self.cancel=threading.Event();self.ready=threading.Event()
        self.waiting=False;self.placement=PLACEMENTS[0];self.session=''
        self.latest=None;self.message='启动 Camera B，请稍候……';self.last_rejection='';self.count=0;self.total=10
        self.done=False;self.closing=False;self.result_status='';self.detected_center=None;self.detected_tags=0
        root.title('Camera B · Registration Field · 仅相机，不连接机器人')
        root.geometry('1160x900')
        self.heading=ttk.Label(root,text='',font=('Noto Sans CJK SC',18,'bold'));self.heading.pack(pady=6)
        self.note=ttk.Label(root,text='先在桌面内旋转板约30°，不要抬高或倾斜；避免横向图案边界的深度空洞。M = 图像中排。',font=('Noto Sans CJK SC',11));self.note.pack()
        self.picture=ttk.Label(root);self.picture.pack(pady=5)
        self.status=ttk.Label(root,text='',wraplength=1100,font=('Noto Sans CJK SC',13));self.status.pack(pady=5)
        self.rejection=ttk.Label(root,text='',wraplength=1100,foreground='#a33',font=('Noto Sans CJK SC',11));self.rejection.pack()
        self.bar=ttk.Progressbar(root,length=900,mode='determinate');self.bar.pack(pady=4)
        controls=ttk.Frame(root);controls.pack(pady=8)
        self.go=ttk.Button(controls,text='板已摆好、手已移开 → 开始当前位置（Enter）',command=self.begin);self.go.pack(side='left',padx=10)
        self.stop=ttk.Button(controls,text='停止并恢复相机设置（Esc）',command=self.abort);self.stop.pack(side='left',padx=10)
        self.report=ttk.Button(controls,text='打开结果目录',command=self.open_report,state='disabled');self.report.pack(side='left',padx=10)
        self.path=ttk.Label(root,text='',wraplength=1100);self.path.pack()
        root.bind('<Return>',lambda e:self.begin());root.bind('<Escape>',lambda e:self.abort())
        root.protocol('WM_DELETE_WINDOW',self.close_request)
        self.photo=None;self.refresh()
    def begin(self):
        if self.waiting and not self.done:self.ready.set();self.waiting=False
    def abort(self):
        self.cancel.set();self.message='正在停止采集并恢复 emitter/曝光设置，请等待……'
    def close_request(self):
        if self.done:self.root.destroy()
        else:self.closing=True;self.abort()
    def update(self,message,count,total):
        self.message=message;self.count=count;self.total=total
        if message.startswith('已拒绝'):self.last_rejection=message
        elif message.startswith('请将'):self.last_rejection=''
    def preview(self,rgb):self.latest=rgb
    def finished(self,status,session,error):
        self.done=True;self.waiting=False;self.result_status=status;self.session=session
        self.message=f'结束状态：{status}。'+(f'原因：{error}' if error else '已生成统计、图和 summary.md。')
    def open_report(self):
        if self.session:webbrowser.open(Path(self.session).resolve().as_uri())
    def refresh(self):
        self.heading.configure(text=f'当前位置 {self.placement}     有效 repeat：{self.count}/{self.total}')
        self.status.configure(text=self.message);self.path.configure(text=self.session)
        self.rejection.configure(text=self.last_rejection)
        self.bar.configure(maximum=max(1,self.total),value=self.count)
        self.go.configure(state='normal' if self.waiting and not self.done else 'disabled')
        if self.latest is not None:
            image=overlay(self.latest,self.placement,self.detected_center if self.waiting else None,self.detected_tags)
            image.thumbnail((1100,620))
            self.photo=ImageTk.PhotoImage(image);self.picture.configure(image=self.photo)
        if self.done:
            self.stop.configure(state='disabled');self.report.configure(state='normal')
            if self.closing:self.root.destroy();return
        self.root.after(120,self.refresh)

def launch(worker):
    root=tk.Tk();ui=Interface(root)
    def wrapped():
        try:worker(ui)
        except BaseException as exc:
            ui.finished('FAILED',ui.session,f'{type(exc).__name__}: {exc}')
    thread=threading.Thread(target=wrapped,daemon=False);thread.start()
    try:root.mainloop()
    finally:
        ui.cancel.set()
        # Non-daemon worker completes restoration before process exit.
        thread.join()
    return 0 if ui.result_status=='COMPLETE' else 1
