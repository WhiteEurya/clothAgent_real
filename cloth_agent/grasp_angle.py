"""Local manual gripper-angle editor; no model invocation."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from .dual_arm.geometry import DualArmError
from .gamepad_collect import save_json
from .experiments.tk_lifecycle import main_thread_window

def transform_for(observation):
    return np.asarray(observation.meta.get('corrected_world_from_camera_affine_mm',
                                          observation.meta['world_from_camera_mm']),dtype=float)


def project(xyz, intrinsic, world_camera):
    camera=np.linalg.inv(world_camera) @ np.r_[xyz,1.]
    if camera[2]<=0:raise DualArmError('preview point is behind camera')
    pixel=intrinsic @ camera[:3]
    return pixel[:2]/pixel[2]


def plane_point(pixel,z,intrinsic,world_camera):
    ray=world_camera[:3,:3] @ np.linalg.solve(intrinsic,np.r_[pixel,1.])
    origin=world_camera[:3,3]
    if abs(ray[2])<1e-8:raise DualArmError('camera ray parallel to grasp plane')
    distance=(z-origin[2])/ray[2]
    if distance<=0:raise DualArmError('grasp plane is behind camera')
    return origin+distance*ray


def image_angle_to_rpy(angle,pixel,xyz,rpy,intrinsic,world_camera,axis_tcp):
    if not np.isfinite(angle) or not 0<=angle<=180:raise ValueError('invalid image angle')
    direction=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))])
    a=plane_point(np.asarray(pixel)-30*direction,xyz[2],intrinsic,world_camera)
    b=plane_point(np.asarray(pixel)+30*direction,xyz[2],intrinsic,world_camera)
    desired=b-a
    opening=Rotation.from_euler('xyz',rpy,degrees=True).apply(axis_tcp)
    if min(np.linalg.norm(desired[:2]),np.linalg.norm(opening[:2]))<1e-6:
        raise DualArmError('opening axis has no horizontal direction')
    delta=np.degrees(np.arctan2(desired[1],desired[0])-np.arctan2(opening[1],opening[0]))
    delta=(delta+90)%180-90  # Symmetric jaws: avoid an unnecessary 180-degree spin.
    result=(Rotation.from_euler('z',delta,degrees=True)*Rotation.from_euler('xyz',rpy,degrees=True)).as_euler('xyz',degrees=True)
    return result,float(delta)


def draw_preview(image,points,rpy,intrinsic,world_camera,axis_tcp,path):
    with Image.open(image) as raw:im=raw.convert('RGB')
    d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    for point,color in zip(points,['#ff5252','#42e5ff']):
        x,y=point['pixel_xy'];d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
        d.text((x+10,y+10),point['name'].upper(),font=font,fill=color,stroke_width=2,stroke_fill='black')
    xyz=np.asarray(points[0]['world_xyz_mm'])
    axis=Rotation.from_euler('xyz',rpy,degrees=True).apply(axis_tcp)
    axis=np.r_[axis[:2]/np.linalg.norm(axis[:2]),0.]
    perpendicular=np.array([-axis[1],axis[0],0.])
    endpoints=[xyz+sign*35*axis for sign in [-1,1]]
    d.line([tuple(project(p,intrinsic,world_camera)) for p in endpoints],fill='#ffdc50',width=3)
    for p in endpoints:
        d.line([tuple(project(p+sign*10*perpendicular,intrinsic,world_camera)) for sign in [-1,1]],fill='#ffdc50',width=6)
    if path is not None:im.save(path)
    return im


@main_thread_window
def confirm_preview(proposal, render, grasp_pixel, *, selection=None):
    """Return the operator's image angle, or None. Rendering never moves hardware."""
    import tkinter as tk
    from PIL import ImageTk, ImageOps
    bg='#0b1220';card='#152238';muted='#9eafc6';white='#eff6ff';cyan='#50d9dd'
    root=tk.Tk();root.title('Arm6 · 选点与抓取方向');root.configure(bg=bg)
    width=min(1420,root.winfo_screenwidth()-60)
    height=min(900,root.winfo_screenheight()-80)
    root.geometry(f'{width}x{height}')
    accepted=[];photos=[]
    # This environment's Tk only exposes bitmap X fonts. Rasterize UI text with
    # the installed CJK font so Chinese labels remain smooth and readable.
    def text_image(text, size, fg, background, max_width=1100):
        font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',size)
        rows=[]
        for paragraph in str(text).split('\n'):
            line=''
            for char in paragraph:
                if line and font.getlength(line+char)>max_width:
                    rows.append(line);line=''
                line+=char
            rows.append(line)
        line_height=round(size*1.5)
        im=Image.new('RGB',(max(1,int(max(font.getlength(row) for row in rows))+4),line_height*len(rows)+2),background)
        draw=ImageDraw.Draw(im)
        for i,row in enumerate(rows):draw.text((2,i*line_height),row,font=font,fill=fg)
        return ImageTk.PhotoImage(im)
    def label(parent, text='', textvariable=None, font=None, fg=white, bg=bg, **kwargs):
        widget=tk.Label(parent,bg=bg,**kwargs)
        size=round((font[1] if font else 11)*1.35)
        def update(*_):
            widget.image=text_image(textvariable.get() if textvariable else text,size,fg,bg)
            widget.configure(image=widget.image)
        if textvariable is not None:textvariable.trace_add('write',update)
        update();return widget
    angle=tk.DoubleVar(value=proposal['angle_image_deg'])
    header=tk.Frame(root,bg=bg);header.pack(fill='x',padx=24,pady=(18,12))
    label(header,text='选点与抓取方向' if selection else '确认抓取方向',font=('Noto Sans CJK SC',22,'bold'),fg=white,bg=bg).pack(anchor='w')
    instruction=tk.StringVar(value='先点击 START，再点击 TARGET；调整角度后点击启动。' if selection else
                             '调整只更新预览。点击底部启动按钮后，机械臂才会运动。')
    label(header,textvariable=instruction,fg=muted,bg=bg,
             font=('Noto Sans CJK SC',11)).pack(anchor='w',pady=(4,0))
    footer=tk.Frame(root,bg=bg);footer.pack(side='bottom',fill='x',padx=24,pady=18)
    body=tk.Frame(root,bg=bg);body.pack(fill='both',expand=True,padx=24)
    side=tk.Frame(body,bg=card,width=350);side.pack(side='right',fill='y',padx=(16,0));side.pack_propagate(False)
    left=tk.Frame(body,bg=card);left.pack(side='left',fill='both',expand=True)
    label(left,text='全局视图   ·   红色 START / 青色 TARGET / 黄色夹爪方向',fg=muted,bg=card,
             font=('Noto Sans CJK SC',10),anchor='w').pack(fill='x',padx=14,pady=12)
    overview=tk.Label(left,bg=card);overview.pack(fill='both',expand=True,padx=10)
    label(left,text='夹指标记为方向示意，不代表实际尺寸或碰撞包络。',fg=muted,bg=card,
             font=('Noto Sans CJK SC',10)).pack(pady=12)
    label(side,text='抓取点局部放大',fg=white,bg=card,font=('Noto Sans CJK SC',12,'bold')).pack(anchor='w',padx=16,pady=(12,8))
    detail=tk.Label(side,bg=card);detail.pack()
    status=tk.StringVar()
    label(side,textvariable=status,fg=cyan,bg=card,font=('Noto Sans CJK SC',12,'bold')).pack(pady=(8,0))
    label(side,text='开合方向：0° 水平 · 90° 竖直',fg=muted,bg=card,font=('Noto Sans CJK SC',10)).pack()
    def redraw(*_):
        current=float(angle.get());im,delta=render(current)
        full=ImageOps.contain(im,(max(100,left.winfo_width()-24),max(100,overview.winfo_height()-8)))
        center=grasp_pixel() if callable(grasp_pixel) else grasp_pixel
        x,y=center if center is not None else (im.width/2,im.height/2)
        # Preserve image scale near the image edge; out-of-frame padding is dark.
        crop=Image.new('RGB',(240,180),bg)
        box=(round(x)-120,round(y)-90,round(x)+120,round(y)+90)
        crop.paste(im.crop(box),(0,0));crop=crop.resize((320,240),Image.Resampling.LANCZOS)
        overview.display_size=full.size;overview.original_size=im.size
        photos[:]=[ImageTk.PhotoImage(full),ImageTk.PhotoImage(crop)]
        overview.configure(image=photos[0]);detail.configure(image=photos[1])
        status.set(f'{current:.0f}°  /  相对原姿态 {delta:+.1f}°')
        if selection is not None:start_button.configure(state='normal' if selection.ready else 'disabled')
        slider.delete('all');position=12+current/180*291
        slider.create_line(12,16,303,16,fill='#334b69',width=5,capstyle='round')
        slider.create_line(12,16,position,16,fill=cyan,width=5,capstyle='round')
        slider.create_oval(position-8,8,position+8,24,fill=cyan,outline='')
    slider=tk.Canvas(side,width=315,height=32,bg=card,highlightthickness=0,cursor='hand2')
    slider.pack(padx=16,pady=4)
    def drag(event):angle.set(round(max(0,min(180,(event.x-12)/291*180))));redraw()
    slider.bind('<Button-1>',drag);slider.bind('<B1-Motion>',drag)
    def adjust(delta):angle.set(max(0,min(180,angle.get()+delta)));redraw()
    controls=tk.Frame(side,bg=card);controls.pack(fill='x',padx=16,pady=4)
    def button(parent,text,command,primary=False):
        color=cyan if primary else '#263b56'
        photo=text_image(text,16,bg if primary else white,color)
        widget=tk.Button(parent,image=photo,command=command,bg=color,
            activebackground=color,relief='flat',borderwidth=0,highlightthickness=0,
            width=photo.width()+28,height=photo.height()+16,cursor='hand2')
        widget.image=photo
        return widget
    button(controls,'−5°',lambda:adjust(-5)).pack(side='left')
    button(controls,'+5°',lambda:adjust(5)).pack(side='left',padx=8)
    def reset():angle.set(proposal['angle_image_deg']);redraw()
    button(controls,'恢复初始',reset).pack(side='right')
    if selection is not None:
        from .collection_graph_check import show_graph_check
        button(header,'检查当前照片是否在 Graph 中',
               lambda:show_graph_check(root,selection,label,button)).place(relx=1,y=0,anchor='ne')
        mode_text=tk.StringVar(value='抓取方式：自动下降并闭合')
        label(side,textvariable=mode_text,fg=cyan,bg=card,font=('Noto Sans CJK SC',11)).pack(pady=(12,3))
        def toggle_mode():
            selection.manual_grasp=not selection.manual_grasp
            mode_text.set('抓取方式：手柄微调高度与夹爪' if selection.manual_grasp else '抓取方式：自动下降并闭合')
            instruction.set('将先到起点上方，等待手柄抓取；抬升后松开控件，按 START 交回自动搬运。' if selection.manual_grasp else
                            '点选起终点、调整角度后，点击启动完成自动抓取搬运。')
        button(side,'切换自动 / 手柄抓取',toggle_mode).pack(pady=(0,4))
        def codex_smooth():
            accepted.append('codex_smooth');root.destroy()
        button(side,'交给 Codex 局部去皱',codex_smooth).pack(pady=(0,6))
        def random_drop():
            accepted.append('random_reset');root.destroy()
        button(side,'抓中间并随机丢下',random_drop).pack(pady=(0,6))
    label(side,text='手动调整 · 无需等待模型',fg=cyan,bg=card,
          font=('Noto Sans CJK SC',11,'bold')).pack(anchor='w',padx=16,pady=(12,5))
    reason=tk.Label(side,bg=card,anchor='nw')
    reason.image=text_image(proposal['reason'],14,muted,card,318)
    reason.configure(image=reason.image);reason.pack(fill='both',expand=True,padx=16,pady=(0,12))
    def accept():
        if selection is not None and not selection.ready:
            instruction.set('请先选好有效的 START 和 TARGET。');return
        accepted.append(float(angle.get()));root.destroy()
    button(footer,'取消本次动作',root.destroy).pack(side='left')
    start_button=button(footer,'确认选点和角度 · 启动搬运',accept,True)
    start_button.pack(side='right')
    if selection is not None:
        start_button.configure(state='disabled')
        def edit(index):
            selection.edit(index);instruction.set('请点击 '+('START' if index==0 else 'TARGET'));redraw()
        button(footer,'重选起点',lambda:edit(0)).pack(side='left',padx=(12,0))
        button(footer,'重选终点',lambda:edit(1)).pack(side='left',padx=(8,0))
        def click(event):
            pixel=display_pixel(event.x,event.y,(overview.winfo_width(),overview.winfo_height()),
                                overview.display_size,overview.original_size)
            if pixel is None:return
            try:
                changed_start=selection.click(pixel)
                if changed_start:
                    proposal['angle_image_deg']=selection.initial_angle();angle.set(proposal['angle_image_deg'])
                instruction.set(selection.message)
            except (DualArmError,ValueError) as exc:instruction.set(str(exc))
            redraw()
        overview.bind('<Button-1>',click)
        overview.configure(cursor='crosshair')
    label(footer,text='Q / E 微调 1°，Esc 取消',bg=bg,fg=muted,font=('Noto Sans CJK SC',10)).pack(side='right',padx=20)
    root.bind('<Escape>',lambda e:root.destroy())
    root.bind('q',lambda e:adjust(-1));root.bind('e',lambda e:adjust(1))
    # No Enter shortcut: point-selection Enter must not authorize dispatch.
    overview.bind('<Configure>',lambda e:redraw())
    root.after(100,redraw);root.mainloop()
    return accepted[0] if accepted else None


def choose_angle(runtime,observation,points,rpy,folder,*,review=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    intrinsic=np.asarray(runtime.camera.intrinsics,dtype=float)
    world_camera=transform_for(observation)
    # Stock xArm jaws open along flange Y; account for configured TCP rotation.
    tcp_rotation=Rotation.from_euler('xyz',runtime.config.arms['left'].tcp_offset[3:],degrees=True)
    axis_tcp=tcp_rotation.inv().apply([0.,1.,0.])
    marked=folder/'selected_points.png'
    with Image.open(observation.image) as raw:im=raw.convert('RGB')
    d=ImageDraw.Draw(im)
    for p,color in zip(points,['red','cyan']):
        x,y=p['pixel_xy'];d.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
    im.save(marked)
    xyz=np.asarray(points[0]['world_xyz_mm'],dtype=float)
    axis=Rotation.from_euler('xyz',rpy,degrees=True).apply(axis_tcp)
    if np.linalg.norm(axis[:2])<1e-6:raise DualArmError('opening axis has no horizontal direction')
    axis=np.r_[axis[:2]/np.linalg.norm(axis[:2]),0.]
    vector=project(xyz+30*axis,intrinsic,world_camera)-project(xyz-30*axis,intrinsic,world_camera)
    initial_angle=float(np.degrees(np.arctan2(vector[1],vector[0]))%180)
    proposal=dict(angle_image_deg=initial_angle,source='manual',
                  reason='初始方向沿用本次抓取的原姿态。拖动滑块或用 Q / E 调整，让夹指尽量避开邻近布片。确认后才会执行；调整预览不会移动机械臂。')
    save_json(folder/'initial_angle.json',proposal)
    def resolve(image_angle):
        return image_angle_to_rpy(image_angle,points[0]['pixel_xy'],points[0]['world_xyz_mm'],
                                  rpy,intrinsic,world_camera,axis_tcp)
    def render(image_angle):
        chosen,delta=resolve(image_angle)
        im=draw_preview(observation.image,points,chosen,intrinsic,world_camera,axis_tcp,None)
        return im,delta
    render(proposal['angle_image_deg'])[0].save(folder/'initial_preview.png')
    decision=dict(operator_accepted=False,initial_angle_image_deg=initial_angle,
                  selection_method='manual',model_calls=0,
                  opening_axis_tcp=axis_tcp.tolist(),camera_intrinsics=intrinsic.tolist())
    save_json(folder/'decision.json',decision)
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled before angle review')
    selected=(review or confirm_preview)(proposal,render,points[0]['pixel_xy'])
    if selected is None:
        raise KeyboardInterrupt('operator cancelled grasp angle preview')
    if runtime.cancel.is_set():raise KeyboardInterrupt('cancelled during angle review')
    # Only the explicitly reviewed final value is sent to motion planning.
    if isinstance(selected,bool):raise ValueError('review must return a numeric angle or None')
    chosen,delta=resolve(float(selected))
    render(float(selected))[0].save(folder/'preview.png')
    decision.update(operator_accepted=True,selected_angle_image_deg=float(selected),
                    manually_adjusted=not np.isclose(float(selected),proposal['angle_image_deg']),
                    rpy_world_deg=chosen.tolist(),yaw_delta_deg=delta)
    save_json(folder/'decision.json',decision)
    return chosen


def display_pixel(x,y,widget_size,display_size,image_size):
    """Undo centered letterboxing and resize before reading measured depth."""
    dx=x-(widget_size[0]-display_size[0])/2
    dy=y-(widget_size[1]-display_size[1])/2
    if not (0<=dx<display_size[0] and 0<=dy<display_size[1]):return None
    return [int(dx*image_size[0]/display_size[0]),int(dy*image_size[1]/display_size[1])]


class PointAngleSelection:
    def __init__(self,runtime,observation,rpy):
        self.observation=observation;self.rpy=rpy;self.points=[None,None];self.next_index=0
        self.intrinsic=np.asarray(runtime.camera.intrinsics,dtype=float)
        self.world_camera=transform_for(observation)
        self.axis_tcp=Rotation.from_euler('xyz',runtime.config.arms['left'].tcp_offset[3:],degrees=True).inv().apply([0,1,0])
        self.spread=runtime.config.limits['max_surface_spread_mm']
        self.message='请点击 START'
        self.manual_grasp=False

    @property
    def ready(self):
        return (all(p is not None for p in self.points) and
                np.linalg.norm(np.asarray(self.points[0]['world_xyz_mm'])[:2]-
                               np.asarray(self.points[1]['world_xyz_mm'])[:2])>=10)

    def edit(self,index):
        self.next_index=index;self.points[index]=None

    def click(self,pixel):
        if self.next_index is None:return False
        xyz=self.observation.sample(pixel,self.spread)
        index=self.next_index
        self.points[index]=dict(name=('start','target')[index],pixel_xy=list(pixel),world_xyz_mm=np.asarray(xyz).tolist())
        self.next_index=next((i for i,p in enumerate(self.points) if p is None),None)
        if self.next_index is not None:self.message='请点击 '+('START' if self.next_index==0 else 'TARGET')
        elif self.ready:self.message='点位已选好。调整夹爪方向，确认后点击启动。'
        else:self.message='两点水平距离不足 10 mm，请重选终点。'
        return index==0

    def initial_angle(self):
        xyz=np.asarray(self.points[0]['world_xyz_mm'])
        axis=Rotation.from_euler('xyz',self.rpy,degrees=True).apply(self.axis_tcp)
        if np.linalg.norm(axis[:2])<1e-6:raise DualArmError('opening axis has no horizontal direction')
        axis=np.r_[axis[:2]/np.linalg.norm(axis[:2]),0.]
        vector=project(xyz+30*axis,self.intrinsic,self.world_camera)-project(xyz-30*axis,self.intrinsic,self.world_camera)
        return float(np.degrees(np.arctan2(vector[1],vector[0]))%180)

    def render(self,image_angle):
        if self.points[0] is not None:
            point=self.points[0]
            chosen,delta=image_angle_to_rpy(image_angle,point['pixel_xy'],point['world_xyz_mm'],self.rpy,
                                          self.intrinsic,self.world_camera,self.axis_tcp)
            return draw_preview(self.observation.image,[p for p in self.points if p is not None],chosen,
                                self.intrinsic,self.world_camera,self.axis_tcp,None),delta
        with Image.open(self.observation.image) as raw:im=raw.convert('RGB')
        if self.points[1] is not None:
            x,y=self.points[1]['pixel_xy'];d=ImageDraw.Draw(im)
            d.ellipse((x-6,y-6,x+6,y+6),outline='#42e5ff',width=3)
        return im,0.


def select_points_and_angle(runtime,observation,rpy,folder,*,review=None):
    selection=PointAngleSelection(runtime,observation,rpy)
    from .collection_graph_check import DEFAULT_GRAPH, DEFAULT_MEMORY
    selection.graph=getattr(runtime,'collection_graph',DEFAULT_GRAPH)
    selection.graph_memory=getattr(runtime,'collection_graph_memory',DEFAULT_MEMORY)
    selection.graph_check_directory=Path(folder).parent/'graph_checks'
    proposal=dict(angle_image_deg=0.,source='manual',reason=
        '在左侧照片中依次点击抓取点和放置点。选好起点后会显示夹爪方向；滑块、±5° 或 Q/E 可调整角度。可随时用底部按钮重选点位。只有最后点击启动才会运动。')
    selected=(review or confirm_preview)(proposal,selection.render,
        lambda:selection.points[0]['pixel_xy'] if selection.points[0] else None,selection=selection)
    if selected is None:raise KeyboardInterrupt('operator cancelled point and angle selection')
    if selected in ('codex_smooth','random_reset'):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
        save_json(folder/'grasp_control.json',dict(mode='codex_smoothing' if selected=='codex_smooth' else 'random_reset',operator_accepted=True))
        return None,rpy,selected
    if not selection.ready:raise DualArmError('both valid points are required before dispatch')
    # Reuse geometry/audit persistence without opening another window.
    chosen=choose_angle(runtime,observation,selection.points,rpy,folder,
                        review=lambda *_:selected)
    save_json(Path(folder)/'grasp_control.json',dict(mode='gamepad' if selection.manual_grasp else 'automatic'))
    return selection.points,chosen,selection.manual_grasp
