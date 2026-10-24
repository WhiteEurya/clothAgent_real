"""Keep short-lived Tcl interpreter finalization on its owning main thread."""
import functools
import gc
import threading


def main_thread_window(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Tk windows must run on the main thread')
        enabled = gc.isenabled()
        # Tk callbacks/widgets form cycles. A later allocation in a recorder or
        # camera thread must not collect the destroyed window's Tcl interpreter.
        gc.disable()
        try:
            return function(*args, **kwargs)
        finally:
            # The function frame (including its widget closures) has unwound.
            # Destroyed interpreters are now collected by their creator thread.
            gc.collect()
            if enabled:
                gc.enable()
    return run
