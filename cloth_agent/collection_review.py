"""Review saved collection RGBs before publishing a trajectory; no hardware IO."""
from pathlib import Path

from .experiments.tk_lifecycle import main_thread_window


@main_thread_window
def confirm_images(folder):
    import tkinter as tk
    from PIL import Image, ImageTk

    folder = Path(folder)
    root = tk.Tk()
    accepted = False
    photos = []
    background = '#101827'

    def finish(value=False):
        nonlocal accepted
        accepted = value
        root.quit()

    try:
        root.title('Arm6 · 操作前后对比 · 确认保存轨迹')
        root.configure(bg=background)
        root.geometry(f'{min(1500, root.winfo_screenwidth()-80)}x'
                      f'{min(900, root.winfo_screenheight()-100)}')
        root.protocol('WM_DELETE_WINDOW', finish)
        root.bind('<Escape>', lambda event: finish())
        tk.Label(root, text='对比操作前后的照片，判断本次抓取是否有效',
                 font=('', 18, 'bold'), bg=background, fg='#eef3fb', pady=14).pack()
        tk.Label(root, text='只有点击“确认保存 traj”才会保存；关闭窗口或 Esc 表示不保存，临时数据保留。',
                 bg=background, fg='#afc0d8').pack()
        controls = tk.Frame(root, bg=background)
        controls.pack(side='bottom', pady=16)
        reject = tk.Button(controls, text='不保存 · 本次无效', command=finish,
                           font=('', 14), padx=18, pady=10)
        reject.pack(side='left', padx=12)
        tk.Button(controls, text='确认保存 traj', command=lambda: finish(True),
                  font=('', 14), bg='#237854', fg='white', padx=18, pady=10).pack(side='left', padx=12)
        canvas = tk.Canvas(root, bg=background, highlightthickness=0)
        canvas.pack(fill='both', expand=True, padx=16, pady=12)
        images = []
        for relative in ('observation/rgb.png', 'after/rgb.png'):
            try:
                with Image.open(folder/relative) as source:
                    images.append((source.convert('RGB'), None))
            except (OSError, ValueError) as exc:
                images.append((None, f'照片不可用：{relative}\n{exc}'))

        def render(event):
            canvas.delete('all')
            photos.clear()
            gap, heading = 16, 40
            width = max(1, (event.width-gap)/2)
            height = max(1, event.height-heading)
            available = [im for im, _ in images if im is not None]
            # One scale for both full frames; never crop or stretch the cloth.
            scale = min([1.] + [min(width/im.width, height/im.height) for im in available])
            for index, ((im, error), title) in enumerate(zip(images, ('操作前 · BEFORE', '操作后 · AFTER'))):
                x = index*(width+gap)+width/2
                canvas.create_text(x, 18, text=title, fill='#eef3fb', font=('', 16, 'bold'))
                if im is None:
                    canvas.create_text(x, heading+height/2, text=error, width=max(1, width-30), fill='#ffc478')
                else:
                    photo = ImageTk.PhotoImage(im.resize(
                        (max(1, round(im.width*scale)), max(1, round(im.height*scale))),
                        Image.Resampling.LANCZOS), master=root)
                    photos.append(photo)
                    canvas.create_image(x, heading+height/2, image=photo)

        canvas.bind('<Configure>', render)
        reject.focus_set()
        root.mainloop()
        return accepted
    finally:
        photos.clear()
        root.destroy()
