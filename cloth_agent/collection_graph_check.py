"""Read-only graph check of the collection UI snapshot, in an isolated process."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT/'reports/arm6_graph_incremental_20261020_v4'
DEFAULT_GRAPH = DEFAULT_BUNDLE/'final/graph.json'
DEFAULT_MEMORY = DEFAULT_BUNDLE/'memory'


class GraphCheck:
    def __init__(self, image, directory, graph=DEFAULT_GRAPH, memory=DEFAULT_MEMORY):
        self.image = Path(image).resolve()
        self.graph = Path(graph).resolve()
        self.memory = Path(memory).resolve()
        self.directory = Path(directory)/datetime.now(timezone.utc).strftime('check_%Y%m%dT%H%M%S%fZ')
        self.process = None
        self.log = None
        self.started = None
        self.cancel_requested = False

    def start(self):
        if self.process is not None:
            raise RuntimeError('graph check already started')
        for path in (self.image, self.graph, self.memory/'memory.json'):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.directory.with_suffix('.log').open('w')
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen([
                sys.executable, '-u', '-m', 'cloth_agent.experiments.retrieval_matcher',
                '--graph', str(self.graph), '--memory', str(self.memory),
                '--image', str(self.image), '--output', str(self.directory),
                '--gpt', '--top-k', '10'], cwd=ROOT, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def cancel(self):
        # SIGINT unwinds the matcher's finally and the Codex backend's subprocess
        # cleanup, including its separate process group. No hardware is involved.
        if not self.cancel_requested and self.process is not None and self.process.poll() is None:
            self.cancel_requested = True
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def close(self):
        self.cancel()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()

    def results(self):
        result = json.loads((self.directory/'result.json').read_text())
        ranking = json.loads((self.directory/'matching/retrieval.json').read_text())['ranking']
        graph = json.loads(self.graph.read_text())
        trajectory = json.loads(Path(graph['source_trajectory']['path']).read_text())
        match = result['match']
        # Put the matched node first for inspection, retaining original ANN ranks.
        cards = [dict(row, ann_rank=i+1,
                      path=trajectory['observations'][row['image_id']]['rgb']['path'])
                 for i, row in enumerate(ranking)]
        if match['status'] == 'MATCH':
            cards.sort(key=lambda row: row['node_id'] != match['node_id'])
        return result, cards


def result_title(match):
    status = match['status']
    if status == 'MATCH':
        return f"匹配到 {match['node_id']} · 置信度 {match.get('confidence', 0):.2f}"
    if status == 'AMBIGUOUS':
        return '匹配有歧义 · 请核对候选照片'
    return '检索候选中未确认匹配 · 不代表全图一定没有'


def show_graph_check(parent, selection, label, button):
    """Modal review prevents starting a motion while a read-only check is pending.

    Poll the subprocess from Tk's owning thread; no Tk objects enter a worker.
    Uses the caller's rasterized CJK label/button factories.
    """
    import tkinter as tk
    from PIL import Image, ImageOps, ImageTk

    check = GraphCheck(selection.observation.image, selection.graph_check_directory,
                       selection.graph, selection.graph_memory)
    window = tk.Toplevel(parent)
    window.title('Arm6 · Graph 状态检查（仅识别）')
    bg = '#0b1220'
    window.configure(bg=bg)
    window.geometry(f'{min(1360, parent.winfo_screenwidth()-80)}x{min(820, parent.winfo_screenheight()-100)}')
    window.transient(parent)
    window.grab_set()
    status = tk.StringVar(value='准备识别当前界面的照片…')
    caption = tk.StringVar(value='左：当前照片     右：图库参考照片')
    label(window, textvariable=status, bg=bg, font=('', 14)).pack(pady=(16, 4))
    label(window, text='当前快照 · DINO Top-10 → GPT · 不生成或执行动作', bg=bg).pack()
    label(window, text=f'Graph：{check.graph.parent.parent.name}', bg=bg).pack(pady=4)
    label(window, textvariable=caption, bg=bg).pack(pady=6)
    controls = tk.Frame(window, bg=bg)
    controls.pack(side='bottom', pady=12)
    canvas = tk.Canvas(window, bg=bg, highlightthickness=0)
    canvas.pack(fill='both', expand=True, padx=16, pady=10)
    photos = []
    cards = []
    images = []
    index = 0
    closing = False
    poll_id = None

    def render(event=None):
        canvas.delete('all')
        photos.clear()
        w, h = max(2, canvas.winfo_width()), max(2, canvas.winfo_height())
        for i, im in enumerate(images):
            shown = ImageOps.contain(im, (max(1, w//2-16), h))
            photo = ImageTk.PhotoImage(shown, master=window)
            photos.append(photo)
            canvas.create_image(w*(.25+.5*i), h/2, image=photo)

    def select(delta=0):
        nonlocal index
        if not cards:
            return
        index = (index+delta) % len(cards)
        card = cards[index]
        with Image.open(card['path']) as source:
            reference = source.convert('RGB')
        images[1:] = [reference]
        caption.set(f"当前照片   |   {card['node_id']} / {card['image_id']} · ANN #{card['ann_rank']}"
                    f" · 相似度 {card['similarity']:.3f} · 候选 {index+1}/{len(cards)}")
        render()

    def close():
        nonlocal closing
        closing = True
        if check.process is not None and check.process.poll() is None:
            status.set('正在取消识别并回收后台进程…')
            check.cancel()
        else:
            window.destroy()
        return 'break'

    def poll():
        nonlocal poll_id, cards
        poll_id = None
        if check.process.poll() is None:
            if not closing:
                status.set(f'正在检索与匹配 · 已等待 {time.monotonic()-check.started:.0f} 秒')
            poll_id = window.after(250, poll)
            return
        if closing:
            window.destroy()
            return
        try:
            if check.process.returncode:
                raise RuntimeError(f'识别进程退出 {check.process.returncode}，日志：{check.directory.with_suffix(".log")}')
            result, cards = check.results()
            status.set(result_title(result['match']))
            select()
            for control in navigation:
                control.configure(state='normal' if cards else 'disabled')
        except Exception as exc:
            status.set(f'识别失败：{exc}')
        print(f'Graph 状态检查记录：{check.directory}', flush=True)

    navigation = [button(controls, '上一张参考', lambda: select(-1)),
                  button(controls, '下一张参考', lambda: select(1))]
    for control in navigation:
        control.pack(side='left', padx=8)
        control.configure(state='disabled')
    button(controls, '关闭 / 取消识别', close).pack(side='left', padx=12)
    window.protocol('WM_DELETE_WINDOW', close)
    window.bind('<Escape>', lambda event: close())
    canvas.bind('<Configure>', render)
    try:
        with Image.open(check.image) as source:
            images.append(source.convert('RGB'))
        try:
            check.start()
            poll_id = window.after(100, poll)
        except Exception as exc:
            status.set(f'无法开始识别：{exc}')
        parent.wait_window(window)
    finally:
        if poll_id is not None:
            try:
                parent.after_cancel(poll_id)
            except tk.TclError:
                pass
        check.close()
        photos.clear()
