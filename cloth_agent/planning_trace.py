"""Capture explicit public planning summaries; never derive private reasoning."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

PREFIX = 'PLANNING_TRACE '
PHASES = {'ORIENT', 'LOCALIZE', 'HYPOTHESIZE', 'IMPLEMENT', 'VERIFY', 'REVISE', 'DONE'}


def trace_prompt():
    return '''
本次同时记录可观察的规划事件摘要。仅输出简洁、可核验的决策摘要，禁止私有 chain-of-thought、逐 token 推理或冗长内心独白。
在实际事件发生时，通过中间 commentary 消息单独输出一行 PLANNING_TRACE 后紧接一个 JSON 对象；不要用命令打印，不要写文件，不要等最终回答再回忆补写。
事件包括：首次理解本轮任务；决定读取/搜索新图像、文件或符号；形成/修改假设；决定修改候选动作代码；运行离线 IK、测试或验证；验证失败；明显改变工作范围；决定本轮规划完成。
每条结构为：{"step":1,"phase":"ORIENT","goal":"具体子目标","working_set":{"files":[],"symbols":[]},"known":[],"unknown":[],"hypothesis":[],"next_action":{"type":"READ","target":"对象","purpose":"目的"},"expected_result":"支持/否定假设的可观察结果","confidence":"MEDIUM"}。
phase 取 ORIENT/LOCALIZE/HYPOTHESIZE/IMPLEMENT/VERIFY/REVISE/DONE；next_action.type 取 READ/SEARCH/EDIT/TEST/COMMAND/OTHER；confidence 取 LOW/MEDIUM/HIGH。step 本轮从1递增，宿主另加全局顺序。
known 只能是已有代码、图像或工具结果支持的事实，推测只写 hypothesis。每条只保留与下一动作相关的摘要；复读同一对象须说明新的信息缺口。不要为了留痕额外调用工具。
实施候选代码前说明当前原因假设、为何改这里、要保持的行为；验证前说明验证目标。工具结果改变理解时追加 REVISE，增加 observed_result 和 state_update，明确被否定假设，不能静默覆盖。
DONE 表示本轮规划结束，不等于机器人已成功执行。DONE 增加 solution、validated_hypotheses、rejected_hypotheses、risks，并给出最终最小 working_set。没有验证就如实说明，不编造测试。
每条字符串最多600字符，每个列表最多8项。记录不替代正常工作，不要求交接记忆。最终 answer 仍严格遵循原动作代码/JSON格式，不包含 PLANNING_TRACE 或整份过程记录。仅中间消息用于留痕，宿主负责按到达顺序落盘。
'''


def validate(entry):
    if not isinstance(entry, dict):
        raise ValueError('entry must be an object')
    required={'step','phase','goal','working_set','known','unknown','hypothesis','next_action','expected_result','confidence'}
    if not required <= entry.keys():
        raise ValueError('missing planning fields')
    if type(entry['step']) is not int or entry['step'] < 1 or entry['phase'] not in PHASES:
        raise ValueError('invalid step or phase')
    if entry['confidence'] not in ('LOW','MEDIUM','HIGH'):
        raise ValueError('invalid confidence')
    w=entry['working_set'];a=entry['next_action']
    if not isinstance(w,dict) or set(w)!={'files','symbols'} or any(not isinstance(w[k],list) for k in w):
        raise ValueError('invalid working_set')
    if not isinstance(a,dict) or set(a)!={'type','target','purpose'} or a['type'] not in ('READ','SEARCH','EDIT','TEST','COMMAND','OTHER'):
        raise ValueError('invalid next_action')
    for k in ('goal','expected_result'):
        if not isinstance(entry[k],str):raise ValueError('invalid summary text')
    for k in ('known','unknown','hypothesis'):
        if not isinstance(entry[k],list):raise ValueError('invalid evidence list')
    if entry['phase']=='REVISE' and not {'observed_result','state_update'} <= entry.keys():
        raise ValueError('REVISE missing observation/update')
    if entry['phase']=='DONE' and not {'solution','validated_hypotheses','rejected_hypotheses','risks'} <= entry.keys():
        raise ValueError('DONE missing conclusion/risks')
    def bounded(value):
        if isinstance(value,str) and len(value)>600:raise ValueError('summary too long')
        if isinstance(value,list):
            if len(value)>8 or any(not isinstance(v,str) for v in value):raise ValueError('invalid summary list')
            for v in value:bounded(v)
        if isinstance(value,dict):
            for v in value.values():bounded(v)
    bounded(entry)
    if len(json.dumps(entry,ensure_ascii=False))>8000:raise ValueError('entry too long')


class TraceCollector:
    """Tail the existing event artifact without changing child stdin/stdout handling."""
    def __init__(self, events, aggregate, local, iteration, session_id=None):
        self.events=Path(events);self.aggregate=Path(aggregate);self.local=Path(local)
        self.iteration=iteration;self.session_id=session_id
        self.stop_event=threading.Event();self.thread=None
        self.status=dict(entries=0,rejected=0,phases=[],errors=[],complete=False,coverage_verified=False)
        self.step=0;self.last_model_step=0;self.seen=set()

    def start(self):
        self.aggregate.parent.mkdir(parents=True,exist_ok=True)
        if self.aggregate.exists():
            for line in self.aggregate.read_text().splitlines():
                if line.strip():self.step=max(self.step,json.loads(line)['step'])
        self.aggregate.touch(exist_ok=True);self.local.touch(exist_ok=True)
        self.thread=threading.Thread(target=self._tail,daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=5)
        if self.thread is not None and self.thread.is_alive():self.status['errors'].append('trace reader did not stop')
        self.status['complete']=('ORIENT' in self.status['phases'] and 'DONE' in self.status['phases']
                                 and not self.status['errors'] and self.status['rejected']==0)
        return self.status

    def _tail(self):
        try:
            with self.events.open() as stream:
                pending='';line_number=0
                while True:
                    chunk=stream.read()
                    pending+=chunk
                    while '\n' in pending:
                        line,pending=pending.split('\n',1);line_number+=1
                        self.consume(line,line_number)
                    if self.stop_event.is_set() and not chunk:
                        if pending:self.consume(pending,line_number+1)
                        return
                    if not chunk:self.stop_event.wait(.02)
        except Exception as exc:
            self.status['errors'].append(f'{type(exc).__name__}: {exc}')

    def consume(self,line,line_number):
        try:event=json.loads(line)
        except (ValueError,TypeError):return
        if event.get('type')=='thread.started':self.session_id=event.get('thread_id')
        item=event.get('item',{})
        # Never copy reasoning events, tool output, or final motion JSON into trace.
        if event.get('type')!='item.completed' or item.get('type')!='agent_message':return
        event_id=item.get('id')
        if event_id is not None and event_id in self.seen:return
        if event_id is not None:self.seen.add(event_id)
        for text in item.get('text','').splitlines():
            if not text.startswith(PREFIX):continue
            try:
                entry=json.loads(text[len(PREFIX):]);validate(entry)
                if entry['step']<=self.last_model_step:raise ValueError('non-increasing model step')
                self.last_model_step=entry['step'];self.step+=1
                entry.update(model_step=entry['step'],step=self.step,iteration=self.iteration,
                    session_id=self.session_id,recorded_at=datetime.now(timezone.utc).isoformat(),
                    source_event_line=line_number,source_item_id=event_id,source='model_public_summary')
                data=json.dumps(entry,ensure_ascii=False)+'\n'
                for path in (self.aggregate,self.local):
                    with path.open('a') as f:f.write(data)
                self.status['entries']+=1;self.status['phases'].append(entry['phase'])
            except (ValueError,TypeError,KeyError) as exc:
                self.status['rejected']+=1
                # Store error category only, not unvalidated or verbose message content.
                if len(self.status['errors'])<8:self.status['errors'].append(str(exc))
