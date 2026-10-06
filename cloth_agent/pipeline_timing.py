"""Optional hierarchical wall-clock timing for the original pipeline."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import csv
from pathlib import Path
import sys
import threading
import time

_current = ContextVar('pipeline_timing', default=None)
_stack = ContextVar('pipeline_timing_stack', default=())


def active_timing():
    return _current.get()


def timed_call(name, fn, *args, **kwargs):
    """Time a synchronous Host operation without changing its arguments/results."""
    recorder = active_timing()
    if recorder is None:
        return fn(*args, **kwargs)
    with recorder.span(name):
        return fn(*args, **kwargs)


def write_host_timeline(directory, spans, elapsed):
    """Partition Host wall time; retain concurrent spans instead of double counting."""
    by_id = {r['id']: r for r in spans}
    def depth(row):
        n = 0
        while row.get('parent_id') in by_id:
            row = by_id[row['parent_id']]
            n += 1
        return n
    boundaries = sorted({0.0, elapsed, *(r['start_s'] for r in spans),
                         *(r.get('end_s', elapsed) for r in spans)})
    timeline = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        mid = (start+end)/2
        active = [r for r in spans if r['start_s'] <= mid < r.get('end_s', elapsed)]
        parents = {r['parent_id'] for r in active}
        leaves = [r for r in active if r['id'] not in parents]
        chosen = max(leaves, key=depth) if leaves else None
        row = {'start_s': start, 'end_s': end, 'duration_s': end-start,
               'stage': chosen['name'] if len(leaves) == 1 else 'CONCURRENT' if leaves else 'UNINSTRUMENTED',
               'span_ids': [r['id'] for r in leaves],
               'iteration': chosen['details'].get('iteration') if chosen else None}
        if timeline and all(timeline[-1][k] == row[k] for k in ('stage', 'span_ids', 'iteration')):
            timeline[-1].update(end_s=end, duration_s=end-timeline[-1]['start_s'])
        else:
            timeline.append(row)
    (directory/'host_timeline.json').write_text(json.dumps(timeline, ensure_ascii=False, indent=2)+'\n')
    with (directory/'host_timeline.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['start_s', 'end_s', 'duration_s', 'stage', 'iteration', 'span_ids'])
        writer.writeheader()
        writer.writerows(timeline)
    with (directory/'host_spans.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'parent_id', 'stage', 'iteration', 'start_s', 'duration_s', 'exclusive_s', 'status', 'details'])
        for r in spans:
            writer.writerow([r['id'], r['parent_id'], r['name'], r['details'].get('iteration'),
                             r['start_s'], r.get('duration_s'), r['exclusive_s'], r['status'],
                             json.dumps(r['details'], ensure_ascii=False)])


def timed_stage(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            recorder = active_timing()
            if recorder is None:
                return fn(*args, **kwargs)
            details = {k: str(v) if isinstance(v,Path) else v for k,v in kwargs.items()
                       if k in {'iteration','stage','label','debug_dir','usage_stage','output_dir'}
                       and isinstance(v,(str,int,float,bool,Path))}
            if args:
                iteration = getattr(args[0], '_active_iteration', None)
                if isinstance(iteration,tuple) and len(iteration)==2:
                    details.setdefault('iteration',iteration[1].get('iteration'))
            with recorder.span(name, **details):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


class PipelineTiming:
    def __init__(self, directory, *, semantic_phases=False):
        self.semantic_phases = bool(semantic_phases)
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.started=time.monotonic()
        self.spans=[]
        self.milestones=[]
        self.lock=threading.RLock()
        self.logging_error=None

    def mark(self, stage, message, *, iteration=None):
        with self.lock:
            row={'event':'milestone', 'stage':stage, 'message':message,
                 'iteration':iteration, 'elapsed_s':time.monotonic()-self.started}
            self.milestones.append(row)
            self._emit(row)

    def _emit(self,event):
        if self.logging_error:
            return
        try:
            with (self.directory/'events.jsonl').open('a') as f:
                f.write(json.dumps(event,ensure_ascii=False)+'\n')
        except OSError as exc:
            # An instrumentation failure must never change robot control flow.
            self.logging_error=str(exc)
            print(f'[timing] recording failed: {exc}',file=sys.stderr)

    @contextmanager
    def span(self,name,**details):
        stack=_stack.get()
        with self.lock:
            if stack:
                parent=self.spans[stack[-1]-1]
                if 'iteration' in parent['details']:
                    details.setdefault('iteration',parent['details']['iteration'])
            row={'id':len(self.spans)+1,'parent_id':stack[-1] if stack else None,
                 'name':name,'start_s':time.monotonic()-self.started,
                 'status':'RUNNING','thread_id':threading.get_ident(),'details':details}
            self.spans.append(row)
            self._emit(dict(row,event='start'))
        token=_stack.set((*stack,row['id']))
        try:
            yield row
        except BaseException as exc:
            row.update(status='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit,GeneratorExit)) else 'ERROR',
                       error_type=type(exc).__name__)
            raise
        else:
            row['status']='RETURNED'
        finally:
            _stack.reset(token)
            with self.lock:
                row['end_s']=time.monotonic()-self.started
                row['duration_s']=row['end_s']-row['start_s']
                self._emit(dict(row,event='end'))

    def __enter__(self):
        self.token=_current.set(self)
        self.stack_token=_stack.set(())
        return self

    def __exit__(self,kind,value,tb):
        _current.reset(self.token)
        _stack.reset(self.stack_token)
        elapsed=time.monotonic()-self.started
        # Union child intervals: simultaneous/nested work is not double counted.
        children={}
        for row in self.spans:
            children.setdefault(row['parent_id'],[]).append(row)
        def covered(rows,start,end):
            intervals=sorted((max(start,r['start_s']),min(end,r.get('end_s',end))) for r in rows)
            total=0.0;cursor=start
            for left,right in intervals:
                if right>max(cursor,left):total+=right-max(cursor,left)
                cursor=max(cursor,right)
            return total
        for row in self.spans:
            end=row.get('end_s',elapsed)
            row['exclusive_s']=max(0,end-row['start_s']-covered(children.get(row['id'],[]),row['start_s'],end))
        result={'status':'ERROR' if kind else 'RETURNED','elapsed_s':elapsed,
                'semantic_phases':self.semantic_phases,
                'logging_error':self.logging_error,'spans':self.spans,
                'milestones':self.milestones,
                'notes':['duration_s includes children; exclusive_s subtracts their interval union.',
                         'RETURNED means a function returned, not garment/task success.',
                         'Uninstrumented work is included in its parent exclusive_s.',
                         'Background recorder threads are included in enclosing wall time, not independently attributed.',
                         'Model event timestamps are saved in each original Claude trace.']}
        try:
            write_host_timeline(self.directory, self.spans, elapsed)
            with (self.directory/'milestones.csv').open('w', encoding='utf-8-sig', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=['elapsed_s','iteration','stage','message'], extrasaction='ignore')
                writer.writeheader()
                writer.writerows(self.milestones)
            calls=[]
            semantic_calls=[]
            for row in self.spans:
                debug=row['details'].get('debug_dir')
                if not debug:
                    continue
                trace=Path(debug)/'claude_stdout.txt'
                if not trace.is_file():
                    if self.semantic_phases:
                        from .semantic_timing import PLANNING_CALLS
                        if row['details'].get('usage_stage') in PLANNING_CALLS:
                            semantic_calls.append({'span_id':row['id'],
                                'stage':row['details'].get('usage_stage'),
                                'iteration':row['details'].get('iteration'),
                                'trace':str(trace),'error':'MISSING_TRACE'})
                    continue
                try:
                    from .harness.planner_profile import analyze_events
                    stdout=trace.read_text()
                    call={'span_id':row['id'],'iteration':row['details'].get('iteration'),
                                  'stage':row['details'].get('usage_stage'),
                                  'trace':str(trace),'events':analyze_events(stdout)}
                    calls.append(call)
                    transport_path=Path(debug)/'timing.json'
                    if transport_path.is_file():
                        try:
                            call['transport']=json.loads(transport_path.read_text())
                        except (ValueError, OSError) as exc:
                            call['transport_error']=str(exc)
                    if self.semantic_phases:
                        from .semantic_timing import PLANNING_CALLS, analyze_semantic_events
                        if call['stage'] in PLANNING_CALLS:
                            semantic_calls.append({k:v for k,v in call.items() if k!='events'})
                            semantic_calls[-1]['semantic']=analyze_semantic_events(stdout,call['events'])
                except Exception as exc:
                    calls.append({'span_id':row['id'],'error':type(exc).__name__})
                    if self.semantic_phases:
                        semantic_calls.append({'span_id':row['id'],
                            'stage':row['details'].get('usage_stage'),'error':type(exc).__name__})
            (self.directory/'claude_calls.json').write_text(json.dumps(calls,ensure_ascii=False,indent=2)+'\n')
            (self.directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            lines=['# 原始流程完整计时','',f'总耗时：{elapsed:.3f} 秒。状态：{result["status"]}。','',
                   '包含耗时含子阶段；自身耗时扣除子阶段覆盖区间。不要把父子行相加。RETURNED 仅表示正常返回，不表示任务成功。','',
                   '| ID / 父 ID | 阶段 | iteration | 开始 s | 包含 s | 自身 s | 状态 |',
                   '|---|---|---|---:|---:|---:|---|']
            for r in self.spans:
                lines.append(f"| {r['id']} / {r['parent_id']} | {r['name']} | {r['details'].get('iteration','')} | {r['start_s']:.3f} | {r.get('duration_s',elapsed-r['start_s']):.3f} | {r['exclusive_s']:.3f} | {r['status']} |")
            lines += ['', '## Claude 逐轮响应', '',
                      '等待包括输入处理、网络及服务排队；输出窗口不是纯推理时间。详细工具参数和 schema 错误见 claude_calls.json。', '',
                      '| 调用 ID | 阶段 | 响应轮次 | 首事件前等待 s | 输出窗口 s | 公开工具 |',
                      '|---|---|---:|---:|---:|---|']
            for call in calls:
                events=call.get('events',{})
                tools={c['id']:c['tool'] for c in events.get('tool_calls',[])}
                for row in events.get('rounds',[]):
                    emission=row['emission_s']
                    lines.append(f"| {call['span_id']} | {call['stage']} | {row['round']} | {row['wait_before_first_event_s']:.3f} | {round(emission,3) if emission is not None else 'UNKNOWN'} | {', '.join(tools[k] for k in row['tools'])} |")
            (self.directory/'summary.md').write_text('\n'.join(lines)+'\n')
            if self.semantic_phases:
                from .semantic_timing import write_semantic_report
                write_semantic_report(self.directory,semantic_calls)
        except OSError as exc:
            print(f'[timing] summary could not be saved: {exc}',file=sys.stderr)
        return False
