"""Tk desktop UI; overlays never enter feature extraction."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font as tkfont

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .dino_correspondence import cosine_colors, draw_result, heatmap_overlay, sha
from .dino_correspondence_tool import ResultWriter


class ImagePanel:
    def __init__(self, parent, title, callback, exact_callback=None):
        frame = ttk.Frame(parent)
        frame.pack(side='left', fill='both', expand=True, padx=5)
        ttk.Label(frame, text=title, font=('', 13, 'bold')).pack(anchor='w', pady=5)
        self.canvas = tk.Canvas(frame, bg='#152131', highlightthickness=0, width=640, height=460)
        self.canvas.pack(fill='both', expand=True)
        self.image = None
        self.mapping = None
        self.canvas.bind('<Configure>', lambda e: self.render())
        self.canvas.bind('<Button-1>', lambda e: self.click(e, callback))
        if exact_callback:
            self.canvas.bind('<Shift-Button-1>', lambda e: self.click(e, exact_callback))

    def set(self, image):
        self.image = image
        self.render()

    def render(self):
        if self.image is None:
            return
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        w, h = self.image.size
        scale = min(cw/w, ch/h)
        rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
        x, y = (cw-rw)/2, (ch-rh)/2
        self.mapping = (x, y, rw/w, rh/h)
        self.photo = ImageTk.PhotoImage(self.image.resize((rw, rh), Image.Resampling.BILINEAR))
        self.canvas.delete('all')
        self.canvas.create_image(x, y, anchor='nw', image=self.photo)

    def click(self, event, callback):
        if self.mapping is None:
            return
        x, y, sx, sy = self.mapping
        point = [(event.x-x)/sx, (event.y-y)/sy]
        if 0 <= point[0] < self.image.width and 0 <= point[1] < self.image.height:
            callback(point)


class App:
    def __init__(self, root, experiment, cases):
        self.root, self.experiment, self.cases = root, experiment, cases
        self.index = 0; self.busy = False; self.point = None; self.vertices = []; self.regions = []
        self.record = None; self.scores = None; self.messages = queue.Queue()
        self.target_point = None
        self.direction = 'A_to_B'
        self.writer = ResultWriter(experiment)
        self.closing = False
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.title('DINO Correspondence · A ↔ B')
        root.geometry('1420x960'); root.minsize(960, 740)
        style = ttk.Style(root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont'):
            tkfont.nametofont(name).configure(size=11)
        style.configure('TButton', padding=(8, 5))
        self.mode = tk.StringVar(value='explore'); self.visibility = tk.StringVar(value='visible')
        self.group = tk.StringVar(value='UNASSIGNED'); self.name = tk.StringVar()
        self.annotator = tk.StringVar(); self.heat = tk.BooleanVar(value=True)
        self.radius = tk.DoubleVar(value=2); self.status = tk.StringVar(value='选择两张干净 RGB 图片，或从测试集加载。')
        self.metrics = tk.StringVar(value='点击 A 查询 B，或点击 B 查询 A')
        self.save_status = tk.StringVar(value='保存队列：空')
        bar = ttk.Frame(root, padding=8); bar.pack(fill='x')
        ttk.Button(bar, text='加载 A / B…', command=self.open_pair).pack(side='left')
        ttk.Button(bar, text='上一对', command=lambda: self.navigate(-1)).pack(side='left', padx=5)
        ttk.Button(bar, text='下一对', command=lambda: self.navigate(1)).pack(side='left')
        self.pair_name = tk.StringVar()
        self.choice = ttk.Combobox(bar, textvariable=self.pair_name, state='readonly', width=34)
        self.choice.pack(side='left', padx=8)
        self.choice.bind('<<ComboboxSelected>>', lambda e: self.load_index(self.choice.current()))
        ttk.Label(bar, text='分组').pack(side='left')
        ttk.Combobox(bar, textvariable=self.group, state='readonly', width=13,
                     values=['UNASSIGNED', 'SELF', 'A', 'B', 'C', 'D']).pack(side='left', padx=5)
        ttk.Label(bar, text='点名称').pack(side='left')
        ttk.Entry(bar, textvariable=self.name, width=18).pack(side='left', padx=5)
        modes = ttk.Frame(root, padding=(8, 0)); modes.pack(fill='x')
        ttk.Radiobutton(modes, text='点击探索：A ↔ B 双向查询', variable=self.mode, value='explore', command=self.reset_point).pack(side='left')
        ttk.Radiobutton(modes, text='盲标评估：选 A → 标 B → 揭晓', variable=self.mode, value='blind', command=self.reset_point).pack(side='left', padx=18)
        ttk.Checkbutton(modes, text='显示热力图', variable=self.heat, command=self.render).pack(side='left')
        ttk.Label(modes, text='NMS 半径（patch）').pack(side='left', padx=(15, 3))
        ttk.Spinbox(modes, from_=0, to=10, increment=.5, textvariable=self.radius, width=5).pack(side='left')
        panels = ttk.Frame(root, padding=(5, 3)); panels.pack(fill='both', expand=True)
        self.a = ImagePanel(panels, 'A · 参考图（探索：点击查询 B）', self.click_a)
        self.b = ImagePanel(panels, 'B · 当前图（探索：点击查询 A）', self.click_b, self.click_target)
        legend_bar = ttk.Frame(root, padding=5); legend_bar.pack(fill='x')
        ttk.Label(legend_bar, text='余弦相似度（固定尺度） −1').pack(side='left')
        gradient = np.tile(np.linspace(-1, 1, 240), (16, 1))
        self.legend = ImageTk.PhotoImage(Image.fromarray(cosine_colors(gradient).astype('uint8')))
        ttk.Label(legend_bar, image=self.legend).pack(side='left', padx=5)
        ttk.Label(legend_bar, text='+1 · 黄色：源点；青色 R：反向位置；1/2/3：目标候选。预测为 patch 中心，分数不是正确概率。').pack(side='left')
        labels = ttk.LabelFrame(root, text='对应区域标注（仅用于评估）', padding=7); labels.pack(fill='x', padx=8)
        for value, text in [('visible', '可见：在 B 上点多边形'), ('invisible', '不可见 / 不存在'), ('uncertain', '人工不确定')]:
            ttk.Radiobutton(labels, text=text, variable=self.visibility, value=value, command=self.visibility_changed).pack(side='left', padx=3)
        ttk.Button(labels, text='封闭区域', command=self.close_region).pack(side='left', padx=4)
        ttk.Button(labels, text='清除区域', command=self.clear_regions).pack(side='left')
        ttk.Label(labels, text='标注者').pack(side='left', padx=(10, 3))
        ttk.Entry(labels, textvariable=self.annotator, width=12).pack(side='left')
        self.reveal_button = ttk.Button(labels, text='保存标签并揭晓', command=self.reveal)
        self.reveal_button.pack(side='right', padx=5)
        point_bar = ttk.Frame(root, padding=(8, 3)); point_bar.pack(fill='x')
        ttk.Label(point_bar, text='盲标时 Shift+点击 B：标记精确对应点 GT（可与区域并用）；仅在能确定同一物理位置时标记。').pack(side='left')
        ttk.Button(point_bar, text='清除 GT 点', command=self.clear_target).pack(side='left', padx=5)
        ttk.Label(root, textvariable=self.metrics, justify='left', wraplength=1380, padding=8,
                  font=('TkFixedFont', 11)).pack(fill='x')
        ttk.Label(root, textvariable=self.status, wraplength=1380, padding=8, foreground='#235789').pack(fill='x')
        save_bar = ttk.Frame(root, padding=(8, 3)); save_bar.pack(fill='x')
        ttk.Label(save_bar, textvariable=self.save_status).pack(side='left')
        ttk.Button(save_bar, text='重试保存', command=self.writer.retry).pack(side='right')
        ttk.Label(root, text=f'自动保存：{experiment.root}  ·  无前景限制；纯补边与图外 patch 中心排除。', padding=6).pack(fill='x')
        self.root.after(50, self.poll)
        self.refresh_cases()
        if cases:
            self.root.after(100, lambda: self.load_index(0))

    def refresh_cases(self):
        self.choice['values'] = [f"{c['pair_id']} · {Path(c['a']).name} → {Path(c['b']).name}" for c in self.cases]

    def open_pair(self):
        if self.busy or self.closing:
            return
        a = filedialog.askopenfilename(title='参考图 A', filetypes=[('RGB images', '*.png *.jpg *.jpeg *.bmp *.tif *.webp'), ('All', '*')])
        if not a:
            return
        b = filedialog.askopenfilename(title='当前图 B', filetypes=[('RGB images', '*.png *.jpg *.jpeg *.bmp *.tif *.webp'), ('All', '*')])
        if not b:
            return
        self.cases.append(dict(pair_id=f'custom_{sha(a)[:10]}_{sha(b)[:10]}', a=a, b=b, group='UNASSIGNED'))
        self.refresh_cases(); self.load_index(len(self.cases)-1)

    def navigate(self, delta):
        if self.cases and not self.busy:
            self.load_index((self.index+delta) % len(self.cases))

    def load_index(self, index):
        if self.busy or self.closing or not 0 <= index < len(self.cases):
            if self.cases:
                self.choice.current(self.index)
            return
        self.busy = True; self.index = index; self.choice.current(index)
        self.point = self.record = self.scores = None; self.vertices = []; self.regions = []
        self.target_point = None
        self.direction = 'A_to_B'
        self.metrics.set('正在加载干净 RGB 和缓存特征…')
        self.status.set('首次加载模型与编码可能需要数秒；后续点击不再运行 DINO。')
        case = self.cases[index]; self.group.set(case.get('group', 'UNASSIGNED'))
        # Capture Tk value on the UI thread, not from the encoder worker.
        group = self.group.get()
        def safe_work():
            try:
                _, timing = self.experiment.load(case['a'], case['b'], case['pair_id'], group)
                self.messages.put(('loaded', timing))
            except Exception as exc:
                self.messages.put(('error', str(exc)))
        threading.Thread(target=safe_work, daemon=True).start()

    def poll(self):
        try:
            kind, value = self.messages.get_nowait()
            self.busy = False
            if kind == 'error':
                self.experiment.pair = None
                self.status.set('加载失败：'+value)
                messagebox.showerror('加载失败', value)
            else:
                self.render()
                self.metrics.set(f"模型首次加载 {value['model_load_s']:.3f}s  |  本次两图加载 {value['pair_load_s']:.3f}s  |  "
                                 f"编码 {value['forward_and_transfer_s']:.3f}s  |  新编码 {value['newly_encoded_images']} 张  |  {value['device']}")
                self.mode_hint()
        except queue.Empty:
            pass
        for query_id, error in self.writer.poll():
            if error:
                self.save_status.set(f'保存失败 {query_id}：{error}。请排除原因后点“重试保存”。')
        if not self.writer.failed:
            self.save_status.set(f'后台保存：{len(self.writer.pending)} 条等待完成' if self.writer.pending else '全部结果已保存')
        if self.closing and not self.writer.pending and not self.busy:
            if self.writer.failed:
                self.closing = False
                messagebox.showerror('尚有结果未保存', '保存失败的结果仍在内存中。请排除原因后点击“重试保存”，再关闭窗口。')
            else:
                self.writer.shutdown()
                self.root.destroy()
                return
        self.root.after(50, self.poll)

    def close(self):
        self.closing = True
        self.status.set('正在等待后台保存完成，随后关闭窗口…')

    def mode_hint(self):
        if self.mode.get() == 'blind':
            exposed = self.experiment.pair_exposed()
            self.status.set('此图片对已有结果曝光：新标签会保存为非盲标，不进入盲标命中率。' if exposed else
                            '先在 A 选点，再在 B 圈合理对应区域（可多个）；或选择不可见 / 不确定。最后点击揭晓。')
        else:
            self.status.set('点击 A 查 B，点击 B 查 A；热力图显示在目标图，结果自动保存。盲标评估仍按 A → B 标注。')

    def reset_point(self):
        if self.busy or self.closing:
            return
        self.point = self.record = self.scores = None; self.vertices = []; self.regions = []
        self.target_point = None
        self.direction = 'A_to_B'
        self.metrics.set('等待点击参考图 A' if self.mode.get() == 'blind' else '点击 A 查询 B，或点击 B 查询 A')
        self.render(); self.mode_hint()

    def click_a(self, point):
        self.select_source(point, 'A_to_B')

    def select_source(self, point, direction):
        if self.busy or self.closing or not self.experiment.pair:
            return
        self.direction = direction
        self.point = point; self.record = self.scores = None
        self.vertices = []; self.regions = []
        self.target_point = None
        if self.mode.get() == 'explore':
            self.run_query(dict(visibility='unlabeled', regions=[], before_prediction=False))
        else:
            self.metrics.set(f'A 源点 ({point[0]:.1f}, {point[1]:.1f})；结果隐藏，等待独立标注 B。')
            self.render(); self.mode_hint()

    def click_b(self, point):
        if self.mode.get() == 'explore':
            self.select_source(point, 'B_to_A')
            return
        if self.busy or self.closing or self.mode.get() != 'blind' or self.point is None or self.record is not None:
            return
        if self.visibility.get() == 'visible':
            self.vertices.append(point); self.render()

    def click_target(self, point):
        if self.busy or self.closing or self.mode.get() != 'blind' or self.point is None or self.record is not None:
            return
        if self.visibility.get() == 'visible':
            self.target_point = point
            self.render()

    def clear_target(self):
        if self.record is None:
            self.target_point = None
            self.render()

    def close_region(self):
        if self.record is None and len(self.vertices) >= 3:
            self.regions.append(self.vertices); self.vertices = []; self.render()

    def clear_regions(self):
        if self.record is None:
            self.regions = []; self.vertices = []; self.render()

    def visibility_changed(self):
        if self.visibility.get() != 'visible':
            self.clear_regions()
            self.clear_target()

    def reveal(self):
        if self.busy or self.closing or self.point is None or self.record is not None or self.mode.get() != 'blind':
            return
        if not self.annotator.get().strip():
            self.status.set('请输入标注者姓名或代号。'); return
        if self.visibility.get() == 'visible' and self.vertices:
            if len(self.vertices) < 3:
                self.status.set('多边形至少需要 3 个点；也可以清除后选择不可见 / 不确定。'); return
            self.close_region()
        self.run_query(dict(visibility=self.visibility.get(), regions=self.regions if self.visibility.get() == 'visible' else [],
            target_xy=self.target_point if self.visibility.get() == 'visible' else None,
            before_prediction=True, annotator=self.annotator.get().strip(),
            provenance='human desktop annotation', note='No executed target grasp used as GT'))

    def run_query(self, label):
        started = time.perf_counter()
        try:
            record, scores = self.experiment.query(self.point, self.radius.get(), label, self.name.get(), self.group.get(),
                                                   direction=self.direction)
            self.record, self.scores = record, scores
            self.render()
            r, e = record['result'], record['evaluation']
            candidates = '   '.join(f"#{c['rank']} ({c['xy'][0]:.1f}, {c['xy'][1]:.1f}) s={c['score']:.4f}" for c in r['candidates'])
            gap = '—' if r['independent_peak_gap'] is None else f"{r['independent_peak_gap']:.4f}"
            cycle = '—' if r['cycle_error_patches'] is None else f"{r['cycle_error_patches']:.2f} patch"
            evaluated = f"标签 {e['visibility']} | Top-1 命中 {e['top1_hit']} | Top-3 命中 {e['top3_hit']} | 区域距离 {e['top1_region_error_px']} px"
            if e['top1_point_error_px'] is not None:
                evaluated += f" | 点误差 {e['top1_point_error_px']:.2f} px"
            evaluated += ' | 盲标' if e['eligible_for_blind_metrics'] else ' | 非盲标'
            direction = record['direction'].replace('_to_', ' → ')
            self.metrics.set(f"{direction}  |  {candidates or r['status']}\n独立峰分差 {gap}  |  反向误差 {cycle}  |  目标单 patch ≈ {r['patch_wh_original_b'][0]:.1f}×{r['patch_wh_original_b'][1]:.1f} 原图像素\n{evaluated}")
            self.root.update_idletasks()
            record['timing']['click_to_draw_s'] = time.perf_counter()-started
            record['timing']['click_to_draw_definition'] = 'callback start through Tk idle render; excludes later artifact saving; no display-vsync measurement'
            self.writer.submit(record, scores)
            self.save_status.set(f'后台保存：{len(self.writer.pending)} 条等待完成')
            self.status.set(f"已显示，后台保存中 {record['query_id']}  |  匹配 {r['matching_s']*1000:.2f} ms  |  点击至绘制 {record['timing']['click_to_draw_s']*1000:.1f} ms。反向一致和高分均不保证部位正确。")
        except Exception as exc:
            self.status.set('未完成：'+str(exc))

    def render(self):
        if self.busy or not self.experiment.pair:
            return
        a, b = [i['image'].copy() for i in self.experiment.pair]
        reverse = self.record is not None and self.record.get('direction') == 'B_to_A'
        if reverse:
            a, b = b, a
        if self.record:
            result = self.record['result']
            a = draw_result(a, result, source=True)
            if self.heat.get():
                b = heatmap_overlay(b, self.scores, self.record['images'][1]['meta'])
            b = draw_result(b, result, self.record['label'])
        else:
            if self.point:
                a = draw_result(a, dict(source_xy=self.point), source=True)
            draw = ImageDraw.Draw(b)
            for poly in self.regions:
                draw.line([tuple(p) for p in poly]+[tuple(poly[0])], fill='#55ff55', width=3)
            if len(self.vertices) > 1:
                draw.line([tuple(p) for p in self.vertices], fill='#55ff55', width=3)
            for x, y in self.vertices:
                draw.ellipse((x-3, y-3, x+3, y+3), fill='#55ff55')
            if self.target_point is not None:
                b = draw_result(b, dict(candidates=[]), dict(target_xy=self.target_point))
        if reverse:
            a, b = b, a
        self.a.set(a); self.b.set(b)


def launch(experiment, cases):
    root = tk.Tk()
    App(root, experiment, cases)
    root.mainloop()
