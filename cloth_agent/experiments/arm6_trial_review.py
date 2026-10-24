"""Fixed graph route validation and explicit operator review; no robot commands."""
import copy
import time
from pathlib import Path

from .discovery_data import save
from .graph_policy import evidence_file
from .graph_minimal_grounding import validate
from .tk_lifecycle import main_thread_window


def fixed_route(graph, templates, edge_ids, max_steps):
    if not edge_ids:
        return []
    ids = edge_ids.split(',')
    if len(set(ids)) != len(ids) or len(ids) > max_steps:
        raise ValueError('route must contain unique edges within max-steps')
    edges = {e['edge_id']: e for e in graph['edges']}
    if any(e not in edges or e not in templates for e in ids):
        raise ValueError('route contains unknown or unavailable action templates')
    route = [edges[e] for e in ids]
    if any(a['target_node'] != b['source_node'] for a, b in zip(route, route[1:])):
        raise ValueError('route is not a continuous graph path')
    return route


@main_thread_window
def review_step(current, reference, proposed, template, expected_node, model_match, folder, *, cancel=None, local_action=False):
    """Confirm the expected state and (when present) edit two current-image pixels.

    Keeps the model's original confidence. Approval is separate human evidence,
    never a fabricated high-confidence model prediction. Closing always cancels.
    """
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = copy.deepcopy(proposed) if proposed is not None else None
    with Image.open(current) as image:
        current_image = image.convert('RGB')
    with Image.open(reference) as image:
        reference_image = image.convert('RGB')
    if result is not None:
        validate(result, current_image.size)
    root = tk.Tk(); root.title('Arm6 | Graph route review'); root.configure(bg='#101827')
    poll_timer = [None]
    def close():
        if poll_timer[0] is not None:
            try:root.after_cancel(poll_timer[0])
            except tk.TclError:pass
            poll_timer[0]=None
        root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    width = min(1480, root.winfo_screenwidth()-60)
    height = min(980, root.winfo_screenheight()-80)
    root.geometry(f'{width}x{height}')
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
    info = Image.new('RGB', (width-40, 138), '#101827'); draw = ImageDraw.Draw(info)
    rows = [f'确认当前衣物属于 {expected_node}；模型判断：{model_match.get("node_id") or model_match.get("status")} / {model_match.get("confidence", 0):.2f}',
            '左：当前照片；右：示范照片。红点是抓取点，青点是目标点。',
            '可重选两点；只迁移这条示范动作。状态不对应或衣物已移动时，请停止。' if result is not None else
            '请确认已到达图中的展开终点；确认结果将标记为人工辅助完成。']
    if local_action:
        rows = [f'局部动作 {expected_node}；模型置信度 {model_match.get("confidence", 0):.2f}；目标：展开',
                '左：当前照片；右：动作示范。红点抓取，青点目标；可重选两点。',
                '请确认局部布边/折层适合该动作；不要求整件衣服属于同一状态。不适用请停止。']
    for i, text in enumerate(rows):draw.text((8, 6+i*40), text, font=font, fill='#edf3ff')
    title = tk.Label(root, bg='#101827'); title.pack(padx=20, pady=10)
    title.image = ImageTk.PhotoImage(info); title.configure(image=title.image)
    footer = tk.Frame(root, bg='#101827'); footer.pack(side='bottom', fill='x', padx=20, pady=12)
    confirmed = tk.BooleanVar(value=False); accepted = []
    mode = tk.StringVar(value='grasp_px' if result is not None else '')
    status = tk.StringVar(value='Confirm local action applicability before continuing.' if local_action else 'Confirm the cloth state before continuing.')
    tk.Label(footer, textvariable=status, bg='#101827', fg='white').pack(anchor='w')
    tk.Checkbutton(footer, text=('Local region supports this demonstrated unfolding action' if local_action else 'Current cloth corresponds to the reference state'),
                   variable=confirmed, bg='#101827', fg='white', selectcolor='#263851').pack(anchor='w', pady=6)
    buttons = tk.Frame(footer, bg='#101827'); buttons.pack(fill='x')
    body = tk.Frame(root, bg='#101827'); body.pack(fill='both', expand=True, padx=20)
    canvas = tk.Canvas(body, bg='#18283f', highlightthickness=0); canvas.pack(side='left', fill='both', expand=True)
    ref_widget = tk.Label(body, bg='#18283f'); ref_widget.pack(side='right', padx=(12,0))
    def marked(image, points):
        image = image.copy(); d = ImageDraw.Draw(image)
        for name, color in [('grasp_px', '#ff5252'), ('target_px', '#42e5ff')]:
            point = points.get(name)
            if point is None:continue
            x,y = point; d.ellipse((x-8,y-8,x+8,y+8),outline=color,width=4)
            d.text((x+12,y+8),'START' if name=='grasp_px' else 'TARGET',font=font,fill=color,stroke_width=2,stroke_fill='black')
        if all(points.get(k) is not None for k in ('grasp_px','target_px')):
            d.line([tuple(points['grasp_px']),tuple(points['target_px'])],fill='#ffe082',width=2)
        return image
    ref_points = {} if template is None else dict(grasp_px=template['reference_start_px'], target_px=template['reference_target_px'])
    reference_preview = marked(reference_image, ref_points)
    reference_preview.thumbnail((max(280,width//3), max(250,height-300)))
    ref_widget.image = ImageTk.PhotoImage(reference_preview); ref_widget.configure(image=ref_widget.image)
    transform = {}
    def redraw(*_):
        shown = marked(current_image, result or {})
        shown.thumbnail((max(1,canvas.winfo_width()),max(1,canvas.winfo_height())))
        x = (canvas.winfo_width()-shown.width)//2; y = (canvas.winfo_height()-shown.height)//2
        transform.update(x=x,y=y,width=shown.width,height=shown.height)
        canvas.image = ImageTk.PhotoImage(shown);canvas.delete('all');canvas.create_image(x,y,anchor='nw',image=canvas.image)
    def click(event):
        if result is None or not transform:return
        x=event.x-transform['x'];y=event.y-transform['y']
        if not (0<=x<transform['width'] and 0<=y<transform['height']):return
        key=mode.get();result[key]=[int(x*current_image.width/transform['width']),int(y*current_image.height/transform['height'])]
        mode.set('target_px' if key=='grasp_px' else 'grasp_px')
        status.set(f'START {result["grasp_px"]} / TARGET {result["target_px"]}; editing {mode.get()}')
        redraw()
    def accept():
        if not confirmed.get():status.set('Confirm local applicability, or press Stop.' if local_action else 'Confirm the cloth state, or press Stop.');return
        if result is not None:
            if any(result[k] is None for k in ('grasp_px','target_px')):
                status.set('Select both START and TARGET first.');return
            validate(result,current_image.size)
            if result['grasp_px']==result['target_px']:
                status.set('START and TARGET must differ.');return
        accepted.append(True);close()
    if result is not None:
        for label,key in [('Edit START','grasp_px'),('Edit TARGET','target_px')]:
            tk.Button(buttons,text=label,command=lambda k=key:mode.set(k),padx=15,pady=10).pack(side='left',padx=4)
    tk.Button(buttons,text=('Confirm action + points' if local_action else 'Confirm state + points') if result is not None else 'Confirm goal',command=accept,padx=20,pady=10).pack(side='left',padx=12)
    tk.Button(buttons,text='Stop',command=close,padx=20,pady=10).pack(side='right')
    canvas.bind('<Configure>',redraw);canvas.bind('<Button-1>',click)
    def poll():
        if cancel is not None and cancel.is_set():close();return
        poll_timer[0]=root.after(100,poll)
    poll_timer[0]=root.after(100,poll)
    try:root.mainloop()
    finally:
        try:close()
        except tk.TclError:pass
        save(folder/'operator_review.json',dict(expected_node=None if local_action else expected_node,
             expected_action=expected_node if local_action else None,approved=bool(accepted),
             current=evidence_file(current),reference=evidence_file(reference),model_match=model_match,
             model_anchors=proposed,selected_anchors=result,points_edited=result!=proposed,
             elapsed_s=time.monotonic()-started,source='human_confirmation',
             review_scope='local_action_applicability' if local_action else 'graph_state_and_anchors'))
    if not accepted:raise KeyboardInterrupt('operator cancelled graph route review')
    marked(current_image,result or {}).save(folder/'approved_preview.png')
    return result
