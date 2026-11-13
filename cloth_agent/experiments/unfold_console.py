"""Small terminal presentation layer; never changes planning or execution."""
from datetime import datetime
from pathlib import Path
import os
import sys
import json
import threading
import time
import statistics
from contextlib import contextmanager

ACTIVE = None


class LiveTerminal:
    """Capture noisy Python/native SDK output; retain a separate terminal FD.

    UI thread only reads logs and updates presentation. It never calls hardware,
    cancels a request, or makes decisions based on estimated time.
    """
    def __init__(self, root):
        self.root=Path(root);self.lock=threading.RLock();self.stop=threading.Event()
        self.state=None;self.model_path=None;self.model_offset=0;self.model_note=''
        self.samples={};self.tick=0;self.last_plain=0
        for run in sorted(self.root.parent.glob('arm7_unfold_*'),reverse=True)[:5]:
            for f in sorted(run.glob('round_*/model_metrics.json'))[-5:]:
                try:
                    value=json.loads(f.read_text(encoding='utf-8')).get('wall_latency_s')
                    if value and 0<float(value)<1800:self.samples.setdefault('model',[]).append(float(value))
                except (OSError,ValueError,TypeError):pass
            try:
                for key,values in json.loads((run/'console_timings.json').read_text()).items():
                    self.samples.setdefault(key,[]).extend(v for v in values if isinstance(v,(int,float)) and 0<v<3600)
            except (OSError,ValueError,TypeError):pass
        self.saved=[os.dup(1),os.dup(2)]
        self.terminal=os.fdopen(os.dup(1),'w',buffering=1,encoding='utf-8',errors='replace')
        self.tty=self.terminal.isatty();self.color=self.tty and 'NO_COLOR' not in os.environ
        self.raw=(self.root/'runtime.log').open('ab',buffering=0)
        self.raw_reader=(self.root/'runtime.log').open('rb');self.last_feedback=''
        self.events=(self.root/'console_events.jsonl').open('a',encoding='utf-8',buffering=1)
        sys.stdout.flush();sys.stderr.flush()
        os.dup2(self.raw.fileno(),1);os.dup2(self.raw.fileno(),2)
        self.thread=threading.Thread(target=self._loop,daemon=True,name='unfold-terminal')
        self.thread.start()

    def write(self,text):
        with self.lock:
            try:
                if self.tty:self.terminal.write('\r\033[2K')
                self.terminal.write(text+'\n');self.terminal.flush()
            except OSError:pass

    def event(self,label,message,warning=False):
        with self.lock:
            try:self.events.write(json.dumps(dict(time=datetime.now().isoformat(),label=label,message=str(message),warning=warning),ensure_ascii=False)+'\n')
            except OSError:pass
            tag=f'{datetime.now():%H:%M:%S}  {label}'
            if self.color:tag=f'\033[{"33" if warning else "36"}m{tag}\033[0m'
            self.write(f'  {tag}  {message}')

    def estimate(self,key):
        values=self.samples.get(key,[])
        if values:return statistics.median(values[-20:]),'历史估计'
        return {'capture':15.,'planning':60.,'model':50.,'motion':60.,'startup':20.,'video':20.,'geometry':5.}.get(key,10.),'初始估计'

    @contextmanager
    def stage(self,label,key):
        estimate,basis=self.estimate(key);state=dict(label=label,key=key,started=time.monotonic(),estimate=estimate,basis=basis)
        with self.lock:previous=self.state;self.state=state;self.last_plain=0
        self.event('进行中',f'{label} · 预计约 {estimate:.0f} 秒（{basis}）')
        success=False
        try:
            yield
            success=True
        finally:
            duration=time.monotonic()-state['started']
            with self.lock:
                self.state=previous
                if success:
                    self.samples.setdefault(key,[]).append(duration)
                    try:(self.root/'console_timings.json').write_text(json.dumps(self.samples,indent=2),encoding='utf-8')
                    except OSError:pass
            self.event('阶段结束' if success else '阶段中断',f'{label} · {duration:.1f} 秒',not success)

    def watch(self,path):
        with self.lock:
            self.model_path=Path(path) if path else None;self.model_offset=0;self.model_note=''

    def read_model_events(self):
        with self.lock:
            if self.model_path is None or not self.model_path.exists():return
            with self.model_path.open('rb') as stream:
                stream.seek(self.model_offset)
                while True:
                    line=stream.readline()
                    if not line or not line.endswith(b'\n'):break
                    self.model_offset=stream.tell()
                    try:event=json.loads(line)
                    except (ValueError,UnicodeError):continue
                    kind=event.get('type')
                    note={'thread.started':'模型会话已建立','turn.started':'模型正在处理图像',
                          'turn.completed':'模型请求已完成，正在检查输出',
                          'turn.failed':'模型请求失败，正在保存原因','error':'模型服务报告错误'}.get(kind)
                    if kind=='item.completed' and event.get('item',{}).get('type')=='agent_message':
                        note='收到动作响应，正在检查完整 JSON'
                    if kind=='error' and str(event.get('message','')).startswith('Reconnecting... '):
                        note='网络连接中断，CLI 正在重连；本次规划尚未完成'
                    # Reasoning text and incomplete action JSON are never displayed or executed.
                    if note and note!=self.model_note:
                        self.model_note=note;self.event('模型流',note,kind in ('turn.failed','error'))

    def read_device_progress(self):
        """Translate existing feedback lines, without issuing device queries."""
        for _ in range(500):
            position=self.raw_reader.tell();line=self.raw_reader.readline()
            if not line or not line.endswith(b'\n'):
                self.raw_reader.seek(position);break
            text=line.decode('utf-8',errors='replace')
            if '[gripper]' not in text:continue
            operation='张开' if '[gripper] open:' in text else '闭合'
            if 'completion confirmed' in text:note=f'夹爪{operation}到位'
            elif 'waiting for measured completion' in text or 'result=WAITING' in text:note=f'等待夹爪{operation}到位反馈'
            else:continue
            if note!=self.last_feedback:
                self.last_feedback=note;self.event('设备进度',note)

    def _loop(self):
        while not self.stop.wait(.5):
            try:
                self.read_model_events()
                self.read_device_progress()
                with self.lock:
                    if not self.state:continue
                    s=self.state;elapsed=time.monotonic()-s['started'];remaining=s['estimate']-elapsed
                    eta=f'预计还需 {remaining:.0f} 秒' if remaining>0 else '已超出估计，仍在等待'
                    text=f'  {"◐◓◑◒"[self.tick%4]} {s["label"]} · 已用 {elapsed:.0f} 秒 · {eta}'
                    if self.tty:
                        self.terminal.write('\r\033[2K'+('\033[36m' if self.color else '')+text+('\033[0m' if self.color else ''));self.terminal.flush()
                    elif elapsed-self.last_plain>=10:
                        self.event('等待',text.strip());self.last_plain=elapsed
                    self.tick+=1
            except Exception:
                # Presentation failures must not alter the robot/request lifecycle.
                import traceback
                traceback.print_exc(file=sys.stderr)
                return

    def close(self):
        self.stop.set();self.thread.join(timeout=2)
        sys.stdout.flush();sys.stderr.flush()
        for target,saved in zip((1,2),self.saved):os.dup2(saved,target);os.close(saved)
        if self.tty:self.terminal.write('\r\033[2K')
        self.terminal.close();self.raw_reader.close();self.raw.close();self.events.close()


def open_session(root):
    global ACTIVE
    ACTIVE=LiveTerminal(root)


def close_session():
    global ACTIVE
    if ACTIVE is not None:
        session=ACTIVE;ACTIVE=None;session.close()


@contextmanager
def stage(label,key):
    if ACTIVE is None:yield
    else:
        with ACTIVE.stage(label,key):yield


def watch_model(path):
    if ACTIVE is not None:ACTIVE.watch(path)


def poll_model():
    if ACTIVE is not None:
        try:ACTIVE.read_model_events()
        except Exception:pass


def write(text):
    if ACTIVE is not None:ACTIVE.write(text)
    else:print(text,flush=True)

SKILLS = {None: 'Simple · 直接抓放', 'garment-tangent-grasp': 'Tangent · 切线展开',
          'garment-chord-grasp': 'Chord · 割线展开',
          'cloth-contour-normal-grasp': 'Normal · 轮廓法线',
          'garment-chord-reorient': 'Reorient · 割线纠偏'}
STATES = {'MODEL_REPORTED_DONE': '模型判断已展开', 'ACTION_LIMIT': '达到动作上限',
          'ABSTAIN': '本轮未提出可执行动作', 'INTERRUPTED': '运行被中断',
          'RESET_FAILED': '初始化失败', 'PLANNING_REJECTED': '重规划后仍未通过',
          'EXECUTION_FAILED': '动作规划或执行失败', 'ERROR': '运行异常',
          'DRY_RUN_OK': '离线启动检查通过'}


def emit(label, message, *, warning=False):
    if ACTIVE is not None:
        ACTIVE.event(label,message,warning);return
    tag=f'{datetime.now():%H:%M:%S}  {label}'
    if sys.stdout.isatty() and 'NO_COLOR' not in os.environ:
        tag=f'\033[{"33" if warning else "36"}m{tag}\033[0m'
    print(f'  {tag}  {message}', flush=True)


def brief(value, limit=180):
    text=' '.join(str(value).split())
    return text if len(text)<=limit else text[:limit-1]+'…'


def selected(value):
    decision=value.get('decision')
    emit('决策', SKILLS.get(value.get('skill'),value.get('skill')) if decision=='ACTION' else decision)
    if value.get('region_name'): emit('部位',brief(value['region_name'],100))
    if value.get('evidence'): emit('理由',brief(value['evidence']))
    if decision=='ACTION' and value.get('skill'):
        emit('几何',f'期望拉距 {value.get("pull_distance_mm")} mm · 正在计算抓放点')


def failure(stage, error, folder, failed_points=None):
    if failed_points:
        detail=f'{len(failed_points)} 个候选缺少有效深度；首个 {failed_points[0].get("role")}={failed_points[0].get("pixel")}'
        emit(stage,detail,warning=True)
        emit('原因',explain(failed_points[0].get('error','')),warning=True)
    else:emit(stage,explain(error),warning=True)
    emit('日志',str(Path(folder)/'skill_failure.json'))


def start(root, goal, limit, reset, planner, reorient):
    write('\n'+'─'*58+'\n  CLOTH / UNFOLD · 双 Arm7\n'+'─'*58)
    emit('运行',Path(root).name)
    emit('目标',str(goal))
    emit('配置',f'最多 {limit} 步 · {planner} · '+('自动初始化' if reset else '保留当前衣物状态'))
    if reorient:emit('纠偏','割线纠偏已加载')
    emit('目录',str(root))


def summary(root, result, elapsed):
    status=result.get('status','UNKNOWN')
    rows=result.get('steps',[])
    completed=sum(r.get('execution',{}).get('status')=='EXECUTED' for r in rows)
    write('\n'+'─'*58)
    emit('结束',f'{STATES.get(status,status)} [{status}]',warning=status not in ('MODEL_REPORTED_DONE','DRY_RUN_OK'))
    emit('统计',f'完成展开动作 {completed} 次 · 耗时 {elapsed/60:.1f} 分钟')
    reason=result.get('error') or result.get('reason')
    if reason:emit('原因',explain(reason),warning=True)
    elif status=='ACTION_LIMIT':emit('原因','已用完本次允许的动作次数，模型尚未报告完成。')
    elif status=='MODEL_REPORTED_DONE':emit('原因','模型判断当前衣物已达到展开要求，主动结束。')
    elif status=='DRY_RUN_OK':emit('原因','配置检查完成；没有连接机器人或请求模型。')
    if status=='INTERRUPTED':emit('提示','未完成步骤不计为成功；实际执行记录见日志。')
    emit('记录',str(Path(root)/'episode_summary.json') if status!='DRY_RUN_OK' else str(Path(root)/'protocol.json'))
    if ACTIVE is not None:emit('详细日志',str(Path(root)/'runtime.log'))
    write('─'*58)


def explain(error):
    text=str(error)
    for marker,message in [('no measured source depth','所选抓放点缺少实测深度，无法生成可执行动作。'),
                           ('PyRoki IK rejected','所选动作路径未能通过 IK 求解。'),
                           ('TCP projection','动作越出当前机械臂允许的工作范围，未下发该动作。'),
                           ('unfold XY translation','抓放距离超出允许范围，重规划后仍未得到有效动作。'),
                           ('local normal ambiguous','所选衣物边界弯曲或形状不明确，无法计算可靠的拉动方向。'),
                           ('controller not stationary/ready','机械臂尚未处于静止且可执行的状态，不能开始动作。'),
                           ('TimeoutExpired','模型请求超时，未获得完整动作。'),
                           ('ConnectionResetError','服务连接被重置，请求未完成。'),
                           ('complete collision path check exceeded','完整路径碰撞检查未能在计算预算内完成。')]:
        if marker in text:return message+' 详细信息已保存到日志。'
    return brief(text,240)
