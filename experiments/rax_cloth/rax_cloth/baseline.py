"""Official workspace/workflow + robot registration. No custom exploration policy."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from .upstream import register_checkout
from .session import TASK,write_json
from .official import ensure_library


def register():
    checkout,registry=register_checkout()
    from .skill_library_compat import install
    install()
    from services.workspace.setup import ROBOT_FACTS
    from .xarm_backend import XArmBackend,NAMES
    registry.BACKENDS['cloth']='rax_cloth.xarm_backend:XArmBackend'
    facts=dict(ROBOT_FACTS['mock'])
    facts.update(name='xArm6',body='single active arm** (arm6, 6 DoF)',
                 status=NAMES['robot_state'],observe=NAMES['observe'],move=NAMES['move_tcp'],
                 gripper=NAMES['gripper'],pixel=NAMES['pixel_to_world'],move_joints=NAMES['move_joints'],
                 tools=tuple(d['name'] for d in XArmBackend.tools(None)),
                 overview_camera='camA',overview='camA on arm6',cameras=('camA',),
                 neutral='no automatic Home or observation-pose motion',
                 motion_example='ctx.value("xarm_move_tcp", arm="arm6", pose_mm_deg=pose, observation_id=observation_id)',
                 grasp_example='ctx.value("xarm_gripper", arm="arm6", target="close", observation_id=observation_id)',
                 read_tools=tuple(NAMES[k] for k in ('robot_state','observe','coordinates','pixel_to_world')),
                 first_reads=(('robot-interface','basic control, frame/units, measured calibration and safety contract'),),
                 safety=('Only complete SAFE proofs allow dispatch; UNKNOWN/UNSAFE refuse. Operator arming is required.',))
    ROBOT_FACTS['cloth']=facts
    return checkout


def configure(root,mode,replay_manifest=None,robot_config=None):
    checkout=register()
    root=Path(root).resolve();os.environ['RAX_ARTIFACT_ROOT']=str(root.parent)
    os.environ['RAX_SKILL_LIBRARY']=str(root.parent/('rax-baseline-real-library' if mode=='real_supervised' else 'rax-baseline-offline-library'))
    from services.controller.config import Configuration
    options={'mode':mode,'robot_scope':'arm6'}
    if mode=='real_supervised':
        if robot_config is None:raise ValueError('real_supervised requires --robot-config')
        options['robot_config']=str(Path(robot_config).resolve())
    elif mode=='replay':
        if replay_manifest is None:raise ValueError('replay manifest required')
        options.update(replay_manifest=str(Path(replay_manifest).resolve()),
                       replay_manifest_sha256=hashlib.sha256(Path(replay_manifest).read_bytes()).hexdigest())
    elif mode!='mock':raise ValueError('simulator not validated')
    description=TASK
    if mode!='real_supervised':
        description+='\nEnvironment: '+mode+'. No physical execution or action-conditioned cloth feedback. Physical success is NOT_EVALUATED. Do not invent outcomes.'
    return checkout,Configuration(root=root,task='tshirt_flattening',task_description=description,
                                    robot='cloth',backend_options=options,observation_profile='rgb-only',subagents=0)


def prepare(config):
    from services.workspace import build,recipe_for
    from services.workspace.setup import Setup
    from services import skill_library
    from .xarm_backend import XArmBackend
    root=config.root;ws=root/'agent-workspace';record=root/'baseline-build.json'
    if (root/'configuration.json').exists() and json.loads((root/'configuration.json').read_text())!=config.to_dict():
        raise ValueError('baseline resume configuration differs')
    if ws.exists():
        if not record.exists():raise ValueError('use a fresh baseline session, not an old adaptation workspace')
        return ws
    root.mkdir(parents=True,exist_ok=True)
    write_json(root/'configuration.json',config.to_dict())
    # Original recipe and universal skills, no fixed number of explore/compile steps.
    built=build(recipe_for(),Setup.from_config(config,gemini=False),ws)
    for directory in ('code','memory','output','runtime/frames','tmp','.pylib'):(ws/directory).mkdir(parents=True,exist_ok=True)
    ensure_library(config);built['skill_library']=skill_library.checkout(config,ws)
    (ws/'TASK.md').write_text(config.task_description+'\n')
    interface=ws/'.agents/skills/robot-interface';interface.mkdir(exist_ok=True)
    (interface/'SKILL.md').write_text('''---
name: robot-interface
description: Arm6 primitive interfaces and safety contract; no cloth strategy.
---
Only arm6 = left, 6 DoF, camA is active. No arm7 controller or camB is connected.
Arm7 is only saved static obstacle geometry, not live feedback; no commands may target it.
World = arm6/link_base. Position mm, Euler xyz degrees, joints degrees. Table contact Z=0; no extra offset.
xarm_observe(camera="camA") captures at the current stationary pose, never moves to a preset.
Supply its observation_id to pixel_to_world and every motion/gripper request. Re-observe after actions.
xarm_move_tcp uses the existing Cartesian planner/IK with its existing detour behavior; it is not guaranteed a straight line.
xarm_move_joints uses existing joint-path planning. xarm_gripper only opens/closes. No gains/config setters exist.
The backend checks observation age, calibration and actual joints, full collision/stop proof and feedback freshness.
Only SAFE dispatches; UNKNOWN/UNSAFE refuse. status.safety is not a certificate for a proposed action.
status.safety=UNKNOWN before a proposed action means no action certificate exists, not a controller fault.
connection_state=not_yet_requested means lazy connection; observe/robot_state requests connect arm6.
The shared skill-library origin is read-only /skill-library-origin in the sandbox. Fetch is allowed;
submit changes through the official skill_library_merge tool, never push directly to the shared origin.
Real actions require the upstream operator arming file, rechecked while planning/executing.
The agent cannot arm; disarm, expiration and faults stop. xarm_hold is always permitted and stops the backend.
At fresh real-backend startup, the host may recover a fault-free STOP using existing guarded controller preparation.
This does not arm motion or clear errors. After a hold/fault/cancellation this backend remains latched;
the agent must not reset faults, restart the backend or retry automatically.
Mock/replay never execute or supply counterfactual feedback. Skills from them are not physically verified.
No human garment strategy or production Cloth Agent prompt is supplied. Query tools/list for actual schemas.
xarm_reset(seed=uint32) invokes the user's existing HOST-OWNED environment reset to create a new starting layout.
It is not a learned manipulation skill or flattening success. Its model calls and recordings are separate reset overhead.
It may move arm6 and requires the same operator authorization and complete safety checks as other motion tools.
Use a fresh camA observation after reset; never assume its physical outcome. Mock/replay refuse execution.
Reset can take minutes; do not issue duplicate resets while one is running. A reset refusal is not permission to bypass authorization.
''')
    # CLI/capability overlay only. Preserve the upstream START_PROMPT and universal skill files.
    if config.backend_options.get('manual_reset'):
        path=interface/'SKILL.md'
        text=path.read_text()
        text=text[:text.index('xarm_reset(seed=uint32)')]+(
            'This session uses the onsite terminal. The operator started the session and host supplies motion permission.\n'
            'Request manual environment reset with request_operator(kind="reset"); wait for the operator reply.\n'
            'During the request motion is disabled; the operator presses Enter after resetting and clearing the workspace.\n'
            'No separate arm request is needed. Automatic xarm_reset is unavailable in this session.\n')
        path.write_text(text)
    agents=ws/'AGENTS.md';original=agents.read_text()
    agents.write_text('# Codex CLI / xArm deployment facts\n\n'
        'This is the official RAX workflow with a Codex CLI transport adaptation. '
        'The upstream workflow below is unchanged. Agent Python is /usr/bin/python3 with NumPy/OpenCV. '
        'Host/robot networks, credentials and SDKs are not directly accessible. '
        'General network package installation is unavailable in this adapter. '
        'MCP and api.runtime.Context use /channels/mcp.sock; RGB files are under /frames. '
        'Backend mode: '+config.backend_options['mode']+'. '
        'In mock/replay, do not claim physical success or merge skills as verified.\n\n'+original)
    with (ws/'MCP_SESSION.md').open('a') as f:
        from types import SimpleNamespace
        f.write('\n## Robot backend schemas\n\n```json\n'+json.dumps(XArmBackend.tools(SimpleNamespace(options=config.backend_options)),indent=2)+'\n```\n')
    built['workflow']='upstream START_PROMPT.md and universal skills; no short integration prompt'
    built['cli_adaptation']='Codex transport/sandbox; no second robot RPC service'
    built['start_prompt_sha256']=hashlib.sha256((ws/'START_PROMPT.md').read_bytes()).hexdigest()
    write_json(record,built)
    return ws
