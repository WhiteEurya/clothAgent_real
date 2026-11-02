"""Explicit host episode initialization using the existing protected reset."""
import json
import threading
import time
from .session import write_json


def run(root,backend,console,stop,deadline,seed,progress):
    if stop.is_set() or time.monotonic()>=deadline:
        raise RuntimeError('session stopped before initial reset')
    path=root/'initial-reset.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['status']=='EXECUTED':return previous
        raise RuntimeError('previous initial reset incomplete; inspect initial-reset.json before a new run')
    write_json(path,{'status':'STARTED','seed':seed,'purpose':'environment_initialization'})
    done=threading.Event();outcome={}
    def work():
        try:outcome['result']=backend._reset({'seed':seed})
        except BaseException as exc:outcome['error']=f'{type(exc).__name__}: {exc}'
        finally:done.set()
    worker=threading.Thread(target=work,daemon=True,name='rax-initial-reset')
    progress.emit('先执行现有机械 reset：抓起、落下并重新观察。可用 Ctrl+C / stop 中止。','initial_reset')
    worker.start()
    next_notice=time.monotonic()+5
    while not done.wait(.2):
        console.poll()
        if time.monotonic()>=next_notice:
            progress.emit('初始机械 reset 仍在执行/规划/视觉判断；本轮剩余 '+str(max(0,int(deadline-time.monotonic())))+' 秒。','initial_reset')
            next_notice=time.monotonic()+5
        if stop.is_set() or time.monotonic()>=deadline:
            backend.hold('initial reset interrupted or session deadline reached')
            write_json(path,{'status':'INTERRUPTED','seed':seed,'task_success':None,
                             'physical_outcome':'unknown; inspect reset/action logs'})
            raise RuntimeError('initial reset interrupted; no Agent started')
    if stop.is_set() or time.monotonic()>=deadline:
        backend.hold('session ended during initial reset')
        write_json(path,{'status':'INTERRUPTED','seed':seed,'task_success':None})
        raise RuntimeError('session ended during initial reset; no Agent started')
    if 'error' in outcome:
        write_json(path,{'status':'FAILED','seed':seed,'error':outcome['error']})
        raise RuntimeError('initial reset failed: '+outcome['error'])
    result=outcome['result'];write_json(path,result)
    if result.get('status')!='EXECUTED':raise RuntimeError('initial reset did not complete')
    progress.emit('机械 reset 执行结束；启动 Agent 读取新图像，不把 reset 计作展平成功。','initial_reset')
    return result
