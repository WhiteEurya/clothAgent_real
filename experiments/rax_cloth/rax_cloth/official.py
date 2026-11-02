"""Offline integration of the pinned, unmodified official real/rainbow components.

Runs only trusted interface probes, not an agent or user-supplied Python program.
Workspace generation, session ownership, MCP dispatch and Context come from RAX.
"""
from __future__ import annotations
import argparse
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from .backend import ClothBackend
from .session import TASK, write_json
from .upstream import ROOT, register_checkout


def configure(root, mode, replay_manifest=None, control_socket=None):
    if mode not in {'mock', 'replay', 'real_supervised'}:
        raise PermissionError('Official integration is offline only')
    if mode=='real_supervised' and control_socket is None:
        raise PermissionError('real_supervised requires a separately started operator control service')
    checkout, _ = register_checkout()
    from services.workspace.setup import ROBOT_FACTS
    # Fill the official robot-neutral template facts; never inherit Rainbow advice.
    facts = dict(ROBOT_FACTS['mock'])
    facts.update(name='dual xArm cloth test backend', status='robot_state', observe='observe',
                 move='move_tcp', gripper='gripper', pixel='pixel_to_world',
                 tools=tuple(t['name'] for t in ClothBackend.tools(None)),
                 overview_camera='camA', overview='recorded camA/camB RGB or synthetic interface fixture',
                 cameras=('camA','camB'), body='system** (6-DoF arm6 and 7-DoF arm7)',
                 move_joints='move_joints', neutral='no automatic Home or preset',
                 motion_example='ctx.value("move_tcp", arm="arm6", pose_mm_deg=pose)',
                 grasp_example='ctx.value("gripper", arm="arm6", target="close")',
                 read_tools=('observe','robot_state','coordinates','pixel_to_world'),
                 safety=('Offline mode: no motion or physical feedback. UNKNOWN forbids execution.',),
                 first_reads=(('robot-interface','tool units, camera ownership and offline limitations'),))
    ROBOT_FACTS['cloth'] = facts
    if mode=='real_supervised':
        facts['safety']=('Onsite LB deadman plus complete existing safety proof required. UNKNOWN forbids execution.',)
    root = Path(root).resolve()
    # Operator-side local artifact paths, never agent inputs or global robot settings.
    os.environ['RAX_ARTIFACT_ROOT'] = str(root.parent)
    os.environ['RAX_SKILL_LIBRARY'] = str(root.parent / ('official-supervised-library' if mode=='real_supervised' else 'official-offline-library'))
    from services.controller.config import Configuration
    options = {'mode':mode}
    if mode=='real_supervised':options['control_socket']=str(Path(control_socket).resolve())
    if mode == 'replay':
        if replay_manifest is None:
            raise ValueError('replay needs a manifest')
        options['replay_manifest'] = str(Path(replay_manifest).resolve())
        options['replay_manifest_sha256'] = hashlib.sha256(Path(replay_manifest).read_bytes()).hexdigest()
    config = Configuration(root=root, task='tshirt_flattening', task_description=TASK,
                           robot='cloth', backend_options=options, observation_profile='rgb-only',
                           subagents=0, session_seconds=28800, max_model_calls=1)
    return checkout, config


def ensure_library(config):
    from services import skill_library
    # Git 2.25 on this host predates `git init -b`. Seed the same empty
    # official library using init + symbolic-ref; checkout/validation/merge
    # continue to be the upstream implementation, with no source patch.
    library=skill_library.library_path(config)
    library.parent.mkdir(parents=True,exist_ok=True)
    with (library.parent/'initialization.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (library/'.git').is_dir():
            library.mkdir(parents=True,exist_ok=True)
            skill_library._git('init','-q',cwd=library)
            skill_library._git('symbolic-ref','HEAD','refs/heads/main',cwd=library)
            skill_library._git('config','user.name','rax harness',cwd=library)
            skill_library._git('config','user.email','harness@rax.local',cwd=library)
            (library/'README.md').write_text(skill_library.README.format(
                robot=config.robot,sections=', '.join(f'`## {s}`' for s in skill_library.SECTIONS),
                run_keys=', '.join(f'`{k}`' for k in skill_library.RUN_KEYS),
                image_kb=skill_library.MAX_IMAGE_BYTES//1000,verified=skill_library.VERIFIED_MIN_RUNS))
            (library/'INDEX.md').write_text(skill_library.INDEX)
            (library/'skills').mkdir(exist_ok=True)
            (library/'skills/.gitkeep').write_text('')
            skill_library._git('add','-A',cwd=library)
            skill_library._git('commit','-q','-m','Initialize empty cloth library (Git 2.25 compatible)',cwd=library)


def prepare(config):
    from services.workspace import build, recipe_for
    from services.workspace.setup import Setup
    from services import skill_library
    class OfflineSetup(Setup):
        @property
        def tools(self):
            # No skill merge in offline mode: synthetic outcomes cannot verify a skill.
            return super().tools if config.backend_options['mode']=='real_supervised' else tuple(x for x in super().tools if x != 'skill_library_merge')
    path = config.root / 'configuration.json'
    if path.exists() and json.loads(path.read_text()) != config.to_dict():
        raise ValueError('official resume configuration differs')
    config.root.mkdir(parents=True, exist_ok=True)
    write_json(path, config.to_dict())
    ws = config.root / 'agent-workspace'
    if ws.exists():
        if not (config.root/'workspace-build.json').exists():
            raise ValueError('incomplete workspace preparation; use a fresh session')
        return ws
    setup = OfflineSetup.from_config(config, gemini=False)
    built = build(recipe_for(), setup, ws)
    for name in ('code','memory','output','runtime/frames','tmp','.pylib'):
        (ws/name).mkdir(parents=True, exist_ok=True)
    ensure_library(config)
    built['skill_library'] = skill_library.checkout(config, ws)
    (ws/'TASK.md').write_text('# Task\n\n'+TASK+'\n')
    (ws/'CLAUDE.md').write_text('@AGENTS.md\n')
    (ws/'.claude').mkdir(exist_ok=True)
    (ws/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)
    overrides=[]
    for filename in ('AGENTS.md','MCP_SESSION.md'):
        file=ws/filename
        original=file.read_text()
        revised=('# Offline mode override — '+config.backend_options['mode']+'\n\n'
                 'No real hardware is connected. Motion requests are NOT_EXECUTED; no cloth simulator exists. '
                 'Replay is historical, not an action-conditioned outcome. Physical success is NOT_EVALUATED. '
                 'The upstream workflow below describes the future supervised setting. '
                 'No agent/model/program sandbox has been launched or verified. '
                 'Only RGB is returned; pixel_to_world is a host-side historical calibration query. '
                 'Installed agent packages and internet access are not yet established. '
                 'skill_library_merge is disabled in this mode; do not mark mock/replay skills verified.\n\n'+original)
        revised=revised.replace('You control a **real ', 'The target hardware is a **')
        revised=revised.replace('- Every motion is physical and irreversible.', '- In future supervised sessions, every executed motion is physical and irreversible.')
        revised=revised.replace('(real hardware).', '(offline backend; no hardware connection).')
        if config.backend_options['mode']=='real_supervised':
            revised=('# Supervised real mode\n\n'
                'A separately started operator service owns the robots. No automatic Home or observation-pose motion. '
                'observe(camera="camA" or "camB") captures at the current stationary pose. '
                'Every motion/gripper and pixel_to_world request requires observation_id from a fresh observe. '
                'Only an onsite operator can hold the LB deadman; the agent cannot arm. '
                'Release, stale feedback, heartbeat loss or safety failure stops and latches the service. '
                'UNKNOWN/UNSAFE never permit execution. No additional table-height offset. '
                'Use tools/list for the current schemas; no direct controller/SDK access.\n\n'+original)
        file.write_text(revised)
        overrides.extend(difflib.unified_diff(original.splitlines(True),revised.splitlines(True),fromfile=filename+'.upstream-rendered',tofile=filename))
    interface=ws/'.agents/skills/robot-interface'
    interface.mkdir(parents=True,exist_ok=True)
    (interface/'SKILL.md').write_text('''---
name: robot-interface
description: Dual xArm basic tool contract and explicit offline limits.
---

arm6 is left, 6 DoF, camA. arm7 is right, 7 DoF, camB; these are robot identifiers,
not image sides. World frame is arm6 link_base, positions mm, Euler xyz degrees,
joints degrees. Table contact Z=0 mm; no additional height correction.
Observe returns RGB plus provenance; robot_state is historical or UNKNOWN.
pixel_to_world uses the existing calibrated geometry; no synthetic geometry.
move_tcp, move_joints and gripper are basic requests; offline they return NOT_EXECUTED.
Only SAFE could permit future supervised execution; UNKNOWN and UNSAFE cannot.
No arm, SDK access, controller tuning or configuration modification tool exists.
No initial garment manipulation strategies are supplied.
''')
    if config.backend_options['mode']=='real_supervised':
        from .supervised import real_tools
        (interface/'SKILL.md').write_text('''---
name: robot-interface
description: Protected dual xArm primitive contract.
---
arm6 = left, 6 DoF, camA; arm7 = right, 7 DoF, camB. These are identifiers, not image sides.
World frame arm6 link_base; positions mm, Euler xyz degrees, joints degrees. Table Z=0; no extra offset.
Observe is stationary capture, never a movement. Choose camera explicitly.
Use the returned observation_id for pixel_to_world and every motion/gripper call.
Re-observe after any action; old observations expire or are invalidated.
Only the onsite operator may authorize via held LB. Never wait/queue a motion for future authorization.
UNKNOWN/UNSAFE refuse execution. Safety failures and stop latch until operator restarts the service.
Tools are basic capabilities, not a garment strategy. Query tools/list for schemas.
''')
        contract_path=ws/'MCP_CONTRACT.json'
        contract=json.loads(contract_path.read_text())
        # Match the actual supervised tool schemas, not the offline template's descriptions.
        contract['supervised_backend_tools']=real_tools()
        write_json(contract_path,contract)
        (ws/'MCP_SESSION.md').write_text('# Supervised robot tools\n\n'
            'Use MCP tools/list for the current schemas. Python: from api.runtime import Context.\n'
            'ctx.value(tool, **arguments) returns decoded JSON. No direct SDK or service access.\n'
            'The operator owns all authorization. No queued execution after a refusal.\n'
            'observe(camera=...) never moves; supply its observation_id for pixel conversion and actions.\n'
            'status/request_operator/skill_library_merge follow the original RAX harness.\n\n'
            '```json\n'+json.dumps(real_tools(),indent=2)+'\n```\n')
    (config.root/'workspace-overrides.diff').write_text(''.join(overrides))
    built['adaptation']='offline facts, mode preamble, robot-interface; no upstream source changes'
    built['final_files']={str(f.relative_to(ws)):hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in ws.rglob('*') if f.is_file() and '.git' not in f.parts and not f.is_symlink()}
    write_json(config.root/'workspace-build.json',built)
    return ws


def smoke(root, mode='mock', replay_manifest=None):
    checkout, config=configure(root,mode,replay_manifest)
    from services.controller.session import Session
    from services.controller.frontend import Frontend
    from services.backends import make_backend
    from auto_research_agent.api.runtime import Context
    session=Session(config)  # Official single-owner lock, before any workspace mutation.
    backend=None
    frontend=None
    worker=None
    stop=threading.Event()
    listener=None
    pipe_handles=[]
    sockpath=config.root/'mcp.sock'
    started=time.monotonic()
    try:
        prepare(config)
        backend=make_backend(config)
        existing=list((config.root/'observations').glob('[0-9]*'))
        backend.sequence=max((int(p.name) for p in existing),default=-1)+1
        frontend=Frontend(session,backend,gemini=None,started_at=time.time())
        frontend.library=False  # Deliberate offline guard; no skill promotion from fixture results.
        if len(str(sockpath).encode())>=104:
            raise ValueError('Unix socket path too long; use a shorter session path')
        sockpath.unlink(missing_ok=True)  # Safe: official session ownership lock is held.
        listener=socket.socket(socket.AF_UNIX)
        try:
            listener.bind(str(sockpath))
        except PermissionError:
            listener.close()
            listener=None
            # A transport-only offline fallback: OS pipes, no network access.
            # Request/call/value and server parsing/dispatch remain upstream.
            request_read,request_write=os.pipe()
            response_read,response_write=os.pipe()
            server_reader=os.fdopen(request_read,'r')
            client_writer=os.fdopen(request_write,'w')
            client_reader=os.fdopen(response_read,'r')
            server_writer=os.fdopen(response_write,'w')
            pipe_handles=[server_reader,client_writer,client_reader,server_writer]
            class PipeContext(Context):
                def __init__(self):
                    self._lock=threading.RLock()
                    self._sequence=0
                    self._reader,self._writer=client_reader,client_writer
                    self.request('initialize',{'protocolVersion':'2025-06-18','capabilities':{},
                                               'clientInfo':{'name':'offline-pipe-probe','version':'1'}})
                def close(self):
                    self._writer.close()
                    self._reader.close()
            context_factory=PipeContext
            transport='OS pipes; official Context methods, adapted transport initialization'
        else:
            os.chmod(sockpath,0o600)
            listener.listen(4)
            context_factory=lambda: Context(str(sockpath))
            transport='real Unix socket MCP'
        errors=[]
        def serve():
            try:
                if listener:
                    frontend.serve_socket(listener,stop)
                else:
                    frontend.serve(server_reader,server_writer)
            except BaseException as exc:
                errors.append(repr(exc))
        worker=threading.Thread(target=serve,daemon=True)
        worker.start()
        # Trusted fixed diagnostic through the unchanged official Python client.
        with context_factory() as ctx:
            tools=ctx.request('tools/list',{})['tools']
            assert 'skill_library_merge' not in {t['name'] for t in tools}
            state=ctx.value('status')
            rgb=ctx.call('observe')
            value=json.loads(rgb['content'][0]['text'])
            action=ctx.value('move_tcp',arm='arm6',pose_mm_deg=[0,0,0,0,0,0])
            assert action['status']=='NOT_EXECUTED' and action['physical_outcome'] is None
            try:
                ctx.call('arm')
            except RuntimeError:
                arm_refused=True
            else:
                raise AssertionError('unexpected arm tool')
            assert not state['robot_status']['hardware_connected']
        if errors:
            raise RuntimeError(errors)
        result=dict(mode=mode,official_commit=json.loads((ROOT/'upstream.json').read_text())['commit'],
                    components=['Configuration','workspace.build','skill_library.checkout','Session','Frontend','api.runtime.Context'],
                    transport=transport,interface_test='passed',observed_provenance=value['provenance'],
                    task_success=None,physical_actions=0,llm_calls=0,arm_refused=arm_refused,
                    elapsed_seconds=time.monotonic()-started,official_source_modified=False)
        write_json(config.root/'official-result.json',result)
        (config.root/'summary.md').write_text('# Official RAX component smoke\n\n'+json.dumps(result,indent=2)+'\n\nNo agent exploration or physical evaluation was performed.\n')
        return result
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)
        if listener:
            listener.close()
        for handle in pipe_handles:
            handle.close()
        sockpath.unlink(missing_ok=True)
        if frontend:
            frontend.close()
        else:
            if backend:
                backend.close()
            session.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--mode',choices=['mock','replay'],default='mock')
    parser.add_argument('--replay-manifest',type=Path)
    args=parser.parse_args()
    print(json.dumps(smoke(args.session,args.mode,args.replay_manifest),indent=2))


if __name__=='__main__':
    main()
