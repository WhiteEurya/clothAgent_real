import gc
import threading

import pytest

from cloth_agent.experiments.tk_lifecycle import main_thread_window


@pytest.mark.parametrize('fail', [False, True])
def test_cycles_finalize_on_creator_thread_before_background_work(fail):
    finalized=[]
    original=gc.isenabled()
    class WidgetCycle:
        def __init__(self):self.cycle=self
        def __del__(self):finalized.append(threading.get_ident())
    @main_thread_window
    def window():
        assert not gc.isenabled()
        widget=WidgetCycle()
        if fail:raise ValueError('closed')
    if fail:
        # Match the production failure handler; release the exception traceback
        # before checking the next main-thread collection.
        try:window()
        except ValueError:pass
        gc.collect()
    else:window()
    assert gc.isenabled()==original
    thread=threading.Thread(target=gc.collect);thread.start();thread.join()
    assert finalized==[threading.get_ident()]


def test_worker_cannot_create_window():
    errors=[]
    @main_thread_window
    def window():raise AssertionError('must not run')
    def worker():
        try:window()
        except RuntimeError as exc:errors.append(str(exc))
    thread=threading.Thread(target=worker);thread.start();thread.join()
    assert errors==['Tk windows must run on the main thread']


def test_destroyed_tk_roots_reclaimed_before_worker_gc():
    import os
    import weakref
    import tkinter as tk
    if not os.environ.get('DISPLAY'):pytest.skip('requires display for hidden Tk test')
    roots=[]
    @main_thread_window
    def window():
        root=tk.Tk();root.withdraw()
        roots.append(weakref.ref(root))
        root.callback_cycle=lambda:root
        label=tk.Label(root,text='offline cleanup regression test');label.pack()
        root.update_idletasks();root.destroy()
    for _ in range(8):
        window()
        assert roots[-1]() is None
        worker=threading.Thread(target=gc.collect);worker.start();worker.join()
