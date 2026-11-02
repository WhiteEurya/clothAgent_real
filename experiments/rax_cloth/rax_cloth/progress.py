"""Bounded, human-readable progress; raw model/tool data stays in existing logs."""
import json
import re
import shlex
import sys
import time
from pathlib import Path
from .session import write_json


def short(value, limit=180):
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(value))
    return ''.join(c if c.isprintable() else ' ' for c in text)[:limit]


def duration(seconds):
    seconds=max(0,int(seconds))
    return f'{seconds//60:02d}:{seconds%60:02d}'


class Progress:
    def __init__(self, root, interval=10, stream=None):
        self.root=Path(root);self.interval=interval
        self.stream=stream if stream is not None else sys.stderr
        self.reader=None;self.pending='';self.relay=None;self.deadline=None
        self.max_seconds=0;self.next_heartbeat=0;self.activity='启动中'
        self.model_signature=None
        self.operator_seen={}

    def operator_requests(self, session):
        """Surface original RAX requests; never answer or arm on the agent's behalf."""
        session=Path(session)
        for path in sorted((session/'operator'/'requests').glob('*.json')):
            if path.name.endswith('.reply.json'):continue
            try:row=json.loads(path.read_text())
            except (OSError,ValueError):continue
            identifier=row.get('id');status=row.get('status')
            if not isinstance(identifier,str) or not re.fullmatch('[0-9a-f]{12}',identifier):continue
            if self.operator_seen.get(identifier)==status:continue
            self.operator_seen[identifier]=status
            if status=='pending':
                self.activity='等待现场操作员：'+short(row.get('message',''),320)
                self.emit(self.activity,'operator_request')
                self.emit('请求 '+identifier+'；回复命令：bash experiments/rax_cloth/scripts/rax_operator.sh reply '
                    '--session '+shlex.quote(str(session))+' --request-id '+identifier+' --text "done"',
                    'operator_reply_hint')
            elif status in ('expired','answered'):
                self.emit('操作员请求 '+identifier+'：'+('已超时' if status=='expired' else '已回复'),'operator_request')

    def emit(self, message, kind='activity'):
        now=time.monotonic()
        data={'unix_s':time.time(),'kind':kind,'activity':short(message,400)}
        suffix=''
        if self.deadline is not None:
            left=max(0,self.deadline-now)
            with self.relay.lock:
                calls=self.relay.state['calls_reserved']
            data.update(elapsed_seconds=max(0,self.max_seconds-left),remaining_seconds=left,
                        calls_reserved=calls,calls_limit=self.relay.max_calls)
            suffix=(f' | 已用 {duration(data["elapsed_seconds"])}'
                    f' | 距时间上限 {duration(left)}'
                    f' | 调用 {calls}/{self.relay.max_calls}')
        print(f'[{time.strftime("%H:%M:%S")}] {data["activity"]}{suffix}',file=self.stream,flush=True)
        with (self.root/'progress.jsonl').open('a') as f:
            f.write(json.dumps(data,ensure_ascii=False)+'\n')
        write_json(self.root/'progress.json',data)

    def start(self, events_path, relay, deadline, max_seconds):
        self.reader=Path(events_path).open(errors='replace')
        self.relay=relay;self.deadline=deadline;self.max_seconds=max_seconds
        self.emit('Agent 已启动；剩余时间为预算倒计时，不是任务完成时间预测')

    def event(self, event):
        if not isinstance(event,dict):return
        typ=event.get('type');item=event.get('item') or {}
        if not isinstance(item,dict):item={}
        it= item.get('type');message=None
        if typ=='thread.started':message='会话已连接：'+short(event.get('thread_id'))
        elif typ in ('error','turn.failed'):message='Agent 错误：'+short(event.get('message') or event.get('error'))
        elif typ=='turn.completed':message='Agent 本轮已完成'
        elif typ in ('item.started','item.completed'):
            prefix='开始' if typ=='item.started' else '完成'
            if it=='command_execution':
                command=str(item.get('command','')).splitlines()[0:1]
                message=prefix+'执行命令：'+short(command[0] if command else '')
                if typ=='item.completed':message+=' (exit='+str(item.get('exit_code'))+')'
            elif it=='mcp_tool_call':message=prefix+' MCP：'+short(item.get('tool','未知工具'))
            elif it=='file_change':message=prefix+'修改代码/文件'
            elif it=='agent_message':message='Agent：'+short(item.get('text',''))
        if message:
            self.activity=message;self.emit(message,'agent_event')

    def tick(self, force=False):
        # Read a bounded amount each poll, never print image payloads/tool outputs.
        if self.reader:
            self.pending+=self.reader.read(262144)
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1)
                try:self.event(json.loads(line))
                except (ValueError,TypeError):continue
        pending=None
        if self.relay:
            with self.relay.lock:
                rows=self.relay.state['requests']
                last=rows[-1] if rows else None
                if last:
                    row={k:last.get(k) for k in ('id','status','phase','http_status','started_unix_s','error_type','elapsed_seconds')}
                else:row=None
            if row:
                signature=(row['id'],row['status'],row['phase'])
                phase={'connect_and_send':'连接/发送请求','response_headers':'等待响应头',
                       'response_stream':'接收模型输出'}.get(row['phase'],'等待模型响应')
                if row['status']=='reserved':
                    pending=f'模型请求 #{row["id"]+1}：{phase}，已等待 {duration(time.time()-row["started_unix_s"])}'
                elif row['status']=='returned':
                    pending=f'模型请求 #{row["id"]+1} 返回 HTTP {row["http_status"]}，耗时 {row["elapsed_seconds"]:.1f}s'
                else:pending=f'模型请求 #{row["id"]+1} 失败：{row["error_type"]}'
                if signature!=self.model_signature:
                    self.emit(pending,'model_event');self.model_signature=signature
                if row['status']!='reserved':pending=None
        if force or time.monotonic()>=self.next_heartbeat:
            self.emit(pending or self.activity,'heartbeat')
            self.next_heartbeat=time.monotonic()+self.interval

    def close(self):
        if self.reader:self.reader.close()
