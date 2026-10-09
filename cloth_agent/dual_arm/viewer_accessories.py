"""Visual-only wrist accessories. Approximate housing/supports, no safety geometry."""
from functools import partial
from pathlib import Path
import numpy as np
import yaml


def load_hand_eye(path):
    from .geometry import transform
    doc=yaml.safe_load(Path(path).read_text())
    if doc.get('translation_units','m') != 'm':
        raise ValueError('Expected metre hand-eye translation')
    if doc.get('camera_mount','link_eef') != 'link_eef':
        raise ValueError('Viewer requires link_eef hand-eye mount')
    return transform(doc['X_CammountCam'])


def add_accessories(server, root, arm_id, hand_eye, gripper_source, add_gripper):
    import trimesh
    import yourdfpy
    import viser.transforms as tf
    holder=server.scene.add_frame(root,show_axes=False)
    def mesh(name,m,color,position=(0,0,0),wxyz=(1,0,0,0)):
        return server.scene.add_mesh_simple(root+'/'+name,m.vertices,m.faces,color=color,position=position,wxyz=wxyz)
    if add_gripper:
        robot=yourdfpy.URDF.load(gripper_source,load_meshes=False,load_collision_meshes=False,
            filename_handler=partial(yourdfpy.filename_handler_magic,dir=gripper_source.parent))
        # Reuse only the gripper subtree, never another arm's link transforms.
        keep={'link_eef'}
        changed=True
        while changed:
            changed=False
            for joint in robot.robot.joints:
                if joint.parent in keep and joint.child not in keep:
                    keep.add(joint.child);changed=True
        for name in keep-{'link_eef'}:
            for i,visual in enumerate(robot.link_map[name].visuals):
                if visual.geometry.mesh is None:continue
                source=visual.geometry.mesh
                shape=trimesh.load(gripper_source.parent/source.filename,force='mesh')
                if source.scale is not None:shape.apply_scale(source.scale)
                origin=np.eye(4) if visual.origin is None else visual.origin
                shape.apply_transform(robot.get_transform(name,'link_eef')@origin)
                mesh(f'gripper/{name}_{i}',shape,(235,235,235) if 'base' in name else (45,48,52))
    camera=server.scene.add_frame(root+'/camera',position=hand_eye[:3,3],
        wxyz=tf.SO3.from_matrix(hand_eye[:3,:3]).wxyz,axes_length=.035,axes_radius=.0015)
    # Optical +Z points out of the front. Body-to-optical offset is approximate.
    body=trimesh.creation.box(extents=[.090,.025,.025])
    server.scene.add_mesh_simple(root+'/camera/body',body.vertices,body.faces,color=(215,220,225),position=(0,0,-.0125))
    face=trimesh.creation.box(extents=[.083,.019,.001])
    server.scene.add_mesh_simple(root+'/camera/face',face.vertices,face.faces,color=(22,25,30),position=(0,0,.0005))
    for i,x in enumerate([-.032,-.008,.032]):
        lens=trimesh.creation.cylinder(radius=.004,height=.0015,sections=24)
        server.scene.add_mesh_simple(root+f'/camera/lens{i}',lens.vertices,lens.faces,color=(65,85,105),position=(x,0,.0015))
    server.scene.add_label(root+'/camera/name',f'Cam {"A" if arm_id=="left" else "B"} · D435 (外壳近似)',position=(0,-.028,0),depth_test=False)
    # Photo-inspired mounting plate + straight support. No CAD dimensions claimed.
    plate=trimesh.creation.annulus(r_min=.025,r_max=.038,height=.005)
    mesh('support/flange_plate',plate,(35,55,70),position=(0,0,.003))
    endpoint=hand_eye[:3,3]+hand_eye[:3,:3]@np.array([0,.0125,-.0125])
    start=np.array([0,0,.003]);direction=endpoint-start
    beam=trimesh.creation.box(extents=[.018,.005,float(np.linalg.norm(direction))])
    alignment=trimesh.geometry.align_vectors([0,0,1],direction)
    mesh('support/beam',beam,(35,55,70),position=(start+endpoint)/2,wxyz=tf.SO3.from_matrix(alignment[:3,:3]).wxyz)
    return holder
