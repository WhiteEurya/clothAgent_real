"""Single-terminal operator UI over original RAX arming and requests."""
import select
import sys
import time
import threading


class OnsiteConsole:
    def __init__(self, root, backend, stop, deadline, progress, stream=None):
        from services import arming
        from services.controller import operator
        self.root=root;self.backend=backend;self.stop=stop;self.deadline=deadline
        self.progress=progress;self.stream=stream or sys.stdin
        self.arming=arming;self.operator=operator;self.current=None
        self.paused=threading.Event()

    def enable(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0 or self.backend.latched or self.backend.closed:
            raise PermissionError('session expired or stopped; cannot enable motion')
        self.arming.write(self.root,True,minutes=remaining/60,
                          by='onsite-launcher',reason='operator started supervised session')

    def pause(self):
        self.paused.set()
        self.arming.write(self.root,False,reason='waiting for onsite operator')

    def wait_for_operator(self):
        # Gate before reserving or sending another model request. The operator
        # terminal and original MCP waiter remain running while inference waits.
        while self.paused.is_set():
            if self.stop.wait(.1) or time.monotonic()>=self.deadline:
                raise PermissionError('session stopped while waiting for operator')

    def accept(self,line):
        if not line or line.strip().lower() in ('stop','q','quit'):
            self.backend.hold('onsite console disconnected or stop requested')
            self.stop.set();return
        if self.current is None:return
        pending={r['id']:r for r in self.operator.pending(self.root)}
        if self.current not in pending:
            self.current=None;return
        self.enable()
        self.operator.reply(self.root,self.current,line.strip() or 'Ready; reset complete and hands clear',
                            by='onsite-console')
        self.paused.clear()
        self.progress.emit('已收到现场确认，继续本轮。','operator_reply')
        self.current=None

    def poll(self):
        rows=self.operator.pending(self.root)
        if rows and self.current!=rows[0]['id']:
            self.current=rows[0]['id']
            self.progress.emit('等待你处理：'+rows[0]['message'],'operator_request')
            self.progress.emit('处理好、手离开工作区后按回车继续；输入 stop 或 Ctrl+C 停止。','operator_reply_hint')
        if select.select([self.stream],[],[],0)[0]:self.accept(self.stream.readline())
