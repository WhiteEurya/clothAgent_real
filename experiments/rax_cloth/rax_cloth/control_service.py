"""Explicit local operator entrypoint. Never launched by an agent or overnight job."""
from __future__ import annotations
import argparse
import hmac
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
from .supervised import NativePort,Supervisor

MAX_PACKET=16384


class ServiceUnavailable(RuntimeError):
    """No operator-owned service is available; never create an agent-side capability."""


class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
    daemon_threads=True
    def __init__(self,path,supervisor,secret):
        self.supervisor=supervisor;self.secret=secret;self.slots=threading.BoundedSemaphore(8)
        super().__init__(str(path),Handler)
    def process_request(self,request,address):
        if not self.slots.acquire(blocking=False):request.close();return
        try:super().process_request(request,address)
        except BaseException:self.slots.release();raise
    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:self.slots.release()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(2)
        try:
            raw=self.rfile.readline(MAX_PACKET+1)
            if len(raw)>MAX_PACKET or not raw.endswith(b'\n'):raise ValueError('invalid packet')
            req=json.loads(raw)
            if not isinstance(req,dict) or not hmac.compare_digest(str(req.get('token','')),self.server.secret):raise PermissionError('operator capability required')
            owner=req.get('owner')
            if not isinstance(owner,str) or not 1<=len(owner)<=100:raise ValueError('session id required')
            op=req.get('op')
            if op=='heartbeat':result=self.server.supervisor.heartbeat(owner)
            elif op=='status':result=self.server.supervisor.status()
            elif op=='call':result=self.server.supervisor.call(owner,req.get('tool'),req.get('arguments'))
            else:raise ValueError('unsupported operation')
            response={'ok':True,'result':result}
        except Exception as exc:response={'ok':False,'error':type(exc).__name__+': '+str(exc)}
        try:self.wfile.write(json.dumps(response).encode()+b'\n')
        except OSError:self.server.supervisor.trip('RPC response connection lost')


class Client:
    """Host-owned capability and heartbeat. Only this host gets the service socket."""
    def __init__(self,path,owner):
        self.path=Path(path).resolve();self.owner=owner
        try:
            self.token=(self.path.parent/'capability').read_text().strip()
            if not self.path.is_socket() or not self.token:raise FileNotFoundError()
        except OSError:
            raise ServiceUnavailable(
                '操作员服务尚未就绪或已经退出：'+str(self.path)+
                '。请先在终端 1 启动 run_control_service.sh，输入 CONNECT，'
                '等待显示“受保护服务已连接”并保持该终端运行，再启动 RAX。'
                '不要手动创建 capability 文件；若服务启动失败，请检查终端 1 的错误。') from None
        self.closed=threading.Event();self.failed=None
        try:self.request('heartbeat',timeout=2)
        except OSError:
            raise ServiceUnavailable('无法连接操作员服务：'+str(self.path)+
                                     '。请检查终端 1 是否仍在运行，以及两边 socket 路径是否一致。') from None
        self.worker=threading.Thread(target=self._beat,daemon=True);self.worker.start()
    def request(self,op,timeout=240,**kwargs):
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(timeout);s.connect(str(self.path))
            s.sendall(json.dumps({'op':op,'owner':self.owner,'token':self.token,**kwargs}).encode()+b'\n')
            with s.makefile('rb') as f:line=f.readline(16*1024*1024)
        if not line.endswith(b'\n'):raise ConnectionError('truncated controller response')
        reply=json.loads(line)
        if not reply['ok']:raise PermissionError(reply['error'])
        return reply['result']
    def _beat(self):
        while not self.closed.wait(.2):
            try:self.request('heartbeat',timeout=.5)
            except Exception as exc:
                self.failed=type(exc).__name__
                try:self.request('call',timeout=2,tool='stop',arguments={})
                except Exception:pass
                return
    def status(self):return self.request('status',timeout=2)
    def call(self,name,args):
        if self.failed and name!='stop':raise PermissionError('protected service heartbeat lost')
        try:return self.request('call',tool=name,arguments=args)
        except Exception:
            self.failed='RPC failed'
            self.closed.set()  # Service watchdog latches STOP when lease expires.
            try:self.request('call',timeout=2,tool='stop',arguments={})
            except Exception:pass
            raise
    def close(self):
        result=None
        try:result=self.request('call',timeout=3,tool='stop',arguments={})
        except Exception:self.failed='stop delivery unconfirmed; lease watchdog must stop service'
        finally:self.closed.set();self.worker.join(timeout=2)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('config/dual_arm.local.json'))
    parser.add_argument('--directory',type=Path,required=True,help='NEW private short path, e.g. /tmp/rax-cloth-control')
    parser.add_argument('--log-dir',type=Path,required=True)
    parser.add_argument('--gamepad',default='auto',help='Existing calibrated /dev/input/event* device; auto requires exactly one')
    parser.add_argument('--connect-real',action='store_true')
    args=parser.parse_args()
    if not args.connect_real:parser.error('operator service requires explicit --connect-real')
    if not sys.stdin.isatty():parser.error('onsite interactive operator terminal required')
    print('连接双臂和相机；不自动 Home、不移动拍摄位、不恢复 STOP。确认硬件急停可用。')
    print('动作授权：摇杆回中并松开所有按键后，持续按住 LB；松开/断连停止；START 急停并锁定。')
    if input('现场操作员输入 CONNECT 确认接入：').strip()!='CONNECT':raise SystemExit('cancelled')
    directory=args.directory.resolve();directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    args.log_dir.mkdir(parents=True,exist_ok=False)
    path=directory/'control.sock';secret=secrets.token_hex(32)
    cap=directory/'capability';cap.write_text(secret);cap.chmod(0o600)
    from cloth_agent.gamepad_teleop import Gamepad,Mapper,InputGuard,Intent,list_gamepads
    pad=None;port=None;supervisor=None;server=None;worker=None;done=threading.Event()
    previous={}
    try:
        device=args.gamepad
        if device=='auto':
            rows=list_gamepads()
            if len(rows)!=1:raise RuntimeError('exactly one gamepad required for auto; specify --gamepad /dev/input/eventN')
            device=rows[0]['path']
        pad=Gamepad(device);mapper=Mapper();guard=InputGuard()
        port=NativePort(args.config.resolve(),args.log_dir.resolve())
        supervisor=Supervisor(port,guard,args.log_dir.resolve())
        server=Server(path,supervisor,secret);path.chmod(0o600)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,lambda *_:done.set())
        print('受保护服务已连接（尚未授权运动）。RAX --control-socket '+str(path),flush=True)
        last_enabled=None
        while not done.wait(.02):
            intent=mapper.update(*pad.sample())
            # This is a deadman only, never teleoperation; non-neutral inputs revoke.
            enabled=intent.enabled and not intent.moving and not intent.gripper
            guard.update(Intent(enabled=enabled,stop=intent.stop))
            if enabled!=last_enabled:
                print('现场授权：'+('LB 持续按住，可请求受保护动作' if enabled else '未授权，运动请求拒绝'),flush=True)
                last_enabled=enabled
            if intent.stop:supervisor.trip('operator START');break
            if supervisor.stopped:break
    finally:
        if supervisor:supervisor.trip('operator terminal closed')
        if server:server.shutdown();server.server_close()
        if worker:worker.join(timeout=2)
        if supervisor:supervisor.close()
        elif port:port.close()
        if pad:pad.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        path.unlink(missing_ok=True);cap.unlink(missing_ok=True)
        try:directory.rmdir()
        except OSError:pass


if __name__=='__main__':main()
