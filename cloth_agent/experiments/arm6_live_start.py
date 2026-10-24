"""Live cloth placement and explicit snapshot recognition before fixed-route replay.

One worker owns camera reads and RGB-D captures. No motion commands are issued.
Recognition uses the same captured observation that is handed to grounding.
"""
import queue
import threading
import time
from pathlib import Path

from .discovery_data import save
from .graph_policy import evidence_file, reliable_match
from .tk_lifecycle import main_thread_window


class PlacementSession:
    def __init__(self, runtime, recognize, graph, expected, threshold, folder):
        self.runtime, self.recognize = runtime, recognize
        self.graph, self.expected, self.threshold = graph, expected, threshold
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.phase = 'live'
        self.observation = self.match = None
        self.attempts = 0

    @property
    def ready(self):
        return (self.phase == 'frozen' and self.match is not None
                and reliable_match(self.match, self.graph, self.threshold)
                and self.match['node_id'] == self.expected)

    def resume(self):
        self.phase = 'live'
        self.observation = self.match = None

    def check(self, captured=None):
        self.phase = 'checking'
        self.observation = self.match = None
        folder = self.folder / f'attempt_{self.attempts:04d}'
        self.attempts += 1
        folder.mkdir()
        started = time.monotonic()
        try:
            self.observation = self.runtime.capture_current(folder / 'observation')
            if captured:
                captured(self.observation)
            self.match = self.recognize(self.observation, folder / 'matching')
            self.phase = 'frozen'
            save(folder / 'result.json', dict(expected_node=self.expected,
                 matching=self.match, ready=self.ready,
                 observation=evidence_file(self.observation.image), elapsed_s=time.monotonic()-started))
        except Exception as exc:
            self.phase = 'error'
            save(folder / 'error.json', dict(error=f'{type(exc).__name__}: {exc}'))
            raise

    def accept(self):
        if not self.ready:
            raise ValueError('current snapshot has not reliably matched the required start state')
        save(self.folder / 'accepted.json', dict(expected_node=self.expected, matching=self.match,
             observation=evidence_file(self.observation.image), attempts=self.attempts))
        return self.observation, self.match


@main_thread_window
def prepare_live_start(runtime, reference, recognize, graph, expected, threshold, folder):
    """Arrange on live RGB; freeze/recognize on click; continue only on expected MATCH."""
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw, ImageFont

    session = PlacementSession(runtime, recognize, graph, expected, threshold, folder)
    commands, events = queue.Queue(), queue.Queue()
    frames = queue.Queue(maxsize=1)
    stopped = threading.Event()
    accepted = []
    failure = []

    def worker():
        try:
            while not stopped.is_set():
                try:
                    command = commands.get(timeout=.01 if session.phase == 'live' else .1)
                except queue.Empty:
                    command = None
                if stopped.is_set():
                    break
                if command == 'check':
                    session.check(lambda obs: events.put(('captured', obs)))
                    events.put(('matched', session.match))
                elif command == 'live':
                    session.resume()
                    events.put(('live', None))
                elif session.phase == 'live':
                    rgb, _ = runtime.camera.read()
                    if frames.full():
                        try: frames.get_nowait()
                        except queue.Empty: pass
                    frames.put_nowait(Image.fromarray(rgb).convert('RGB'))
        except Exception as exc:
            events.put(('error', exc))

    root = tk.Tk()
    root.title(f'Arm6 | Live placement -> {expected}')
    root.configure(bg='#101827')
    width = min(1480, root.winfo_screenwidth()-60)
    height = min(980, root.winfo_screenheight()-80)
    root.geometry(f'{width}x{height}')
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 21)
    header = tk.Label(root, bg='#101827'); header.pack(fill='x', padx=16, pady=12)
    def message(text):
        banner = Image.new('RGB', (width-32, 110), '#101827')
        draw = ImageDraw.Draw(banner)
        draw.text((8, 4), f'左侧实时相机  /  右侧目标参考 {expected}', font=font, fill='#67e8f9')
        # Wrap by measured glyph width so model Chinese evidence remains readable.
        line, lines = '', []
        for char in text:
            if draw.textlength(line+char, font=font) > width-60:
                lines.append(line); line = ''
            line += char
        lines.append(line)
        for i, line in enumerate(lines[:2]):
            draw.text((8, 40+i*29), line, font=font, fill='white')
        header.image = ImageTk.PhotoImage(banner); header.configure(image=header.image)

    footer = tk.Frame(root, bg='#101827'); footer.pack(side='bottom', fill='x', padx=18, pady=14)
    body = tk.Frame(root, bg='#101827'); body.pack(fill='both', expand=True, padx=18)
    body.columnconfigure(0, weight=1, uniform='image'); body.columnconfigure(1, weight=1, uniform='image')
    body.rowconfigure(0, weight=1)
    left = tk.Canvas(body, bg='#18283f', highlightthickness=0); left.grid(row=0,column=0,sticky='nsew',padx=(0,6))
    right = tk.Canvas(body, bg='#18283f', highlightthickness=0); right.grid(row=0,column=1,sticky='nsew',padx=(6,0))
    with Image.open(reference) as im: reference_image = im.convert('RGB')
    shown = [None]
    def draw(canvas, image):
        if image is None: return
        image = image.copy(); image.thumbnail((max(1,canvas.winfo_width()),max(1,canvas.winfo_height())))
        canvas.image = ImageTk.PhotoImage(image)
        canvas.delete('all'); canvas.create_image(canvas.winfo_width()//2,canvas.winfo_height()//2,image=canvas.image)
    left.bind('<Configure>', lambda _: draw(left, shown[0]))
    right.bind('<Configure>', lambda _: draw(right, reference_image))
    busy = [False]
    closing = [False]
    def check():
        if busy[0] or closing[0]: return
        busy[0] = True
        check_button.configure(state='disabled'); resume_button.configure(state='disabled'); continue_button.configure(state='disabled')
        message('正在拍摄并识别；请移开双手并保持衣物不动。画面冻结后，识别可能需要几十秒。')
        commands.put('check')
    def resume():
        if busy[0] or closing[0]: return
        busy[0] = True
        continue_button.configure(state='disabled'); resume_button.configure(state='disabled'); check_button.configure(state='disabled')
        commands.put('live')
    def close():
        if closing[0]: return
        closing[0] = True; stopped.set()
        for button in (check_button,resume_button,continue_button): button.configure(state='disabled')
        message('正在停止；等待当前拍摄或识别结束，不会执行抓取。')
    def accept():
        if busy[0] or closing[0] or not session.ready: return
        accepted.append(session.accept()); close()
    check_button = tk.Button(footer,text='Recognize / 识别',command=check,padx=16,pady=12)
    check_button.pack(side='left',padx=5)
    resume_button = tk.Button(footer,text='Resume live / 继续摆放',command=resume,state='disabled',padx=16,pady=12)
    resume_button.pack(side='left',padx=5)
    continue_button = tk.Button(footer,text='Continue / 确认起点',command=accept,state='disabled',padx=16,pady=12)
    continue_button.pack(side='left',padx=5)
    tk.Button(footer,text='Stop',command=close,padx=20,pady=12).pack(side='right')
    root.protocol('WM_DELETE_WINDOW',close)
    message('请参照右图摆放衣物；摆好并移开双手后点击“识别”。匹配成功后确认起点，再进入抓取点预览。')
    thread = threading.Thread(target=worker,name='arm6-live-placement',daemon=True)
    thread.start()
    timer = [None]
    def poll():
        if runtime.cancel.is_set(): close()
        while not events.empty():
            kind, value = events.get_nowait()
            if closing[0]: continue
            if kind == 'captured':
                with Image.open(value.image) as im: shown[0] = im.convert('RGB')
                draw(left,shown[0])
            elif kind == 'matched':
                busy[0] = False
                resume_button.configure(state='normal')
                continue_button.configure(state='normal' if session.ready else 'disabled')
                text = f'{value["status"]} / {value.get("node_id") or "无匹配"} / {value.get("confidence",0):.2f}。'
                text += '起点匹配成功。请勿再动衣物；点击“确认起点”继续。' if session.ready else '尚未匹配目标起点；点击“继续摆放”后再识别。'
                message(text)
            elif kind == 'live':
                busy[0] = False; check_button.configure(state='normal')
                message('实时画面已恢复，旧识别结果已作废。摆好后移开双手，再点击“识别”。')
            elif kind == 'error':
                failure.append(value); close()
        if session.phase == 'live' and not busy[0] and not closing[0]:
            try: shown[0] = frames.get_nowait(); draw(left,shown[0])
            except queue.Empty: pass
        if closing[0] and not thread.is_alive(): root.destroy(); return
        timer[0] = root.after(50,poll)
    timer[0] = root.after(50,poll)
    try:
        root.mainloop()
    finally:
        stopped.set()
        # Never close the shared runtime camera while a capture still owns it.
        thread.join()
        try:
            if timer[0] is not None: root.after_cancel(timer[0])
            root.destroy()
        except tk.TclError: pass
    if failure: raise failure[0]
    if not accepted: raise KeyboardInterrupt('operator cancelled live start placement')
    return accepted[0]
