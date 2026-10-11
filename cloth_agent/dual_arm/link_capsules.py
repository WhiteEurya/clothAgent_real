"""Fixed-radius capsule covers of URDF link geometry.

Tile an expanded PCA-aligned bounding box, not merely its mesh vertices.
Each rectangular prism is contained in a capsule along its long axis, so
triangles and link interiors remain covered between vertices as well.
"""
from pathlib import Path
import numpy as np


def cover_box(vertices, name, radius_mm, shell_mm=2.):
    vertices=np.asarray(vertices,dtype=float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<3 or not np.isfinite(vertices).all():
        raise ValueError('finite 3D mesh vertices required')
    if not np.isfinite(radius_mm) or radius_mm<=0 or not np.isfinite(shell_mm) or shell_mm<0:
        raise ValueError('positive radius and nonnegative shell required')
    center=vertices.mean(axis=0)
    _,_,basis=np.linalg.svd(vertices-center,full_matrices=False)
    local=(vertices-center)@basis.T
    low=local.min(axis=0)-shell_mm;high=local.max(axis=0)+shell_mm
    counts=np.maximum(1,np.ceil((high[1:]-low[1:])/(radius_mm*np.sqrt(2)*(1-1e-10))).astype(int))
    widths=(high[1:]-low[1:])/counts
    capsules=[]
    for i in range(counts[0]):
        for j in range(counts[1]):
            transverse=low[1:]+(np.array([i,j])+.5)*widths
            a=np.r_[low[0],transverse];b=np.r_[high[0],transverse]
            index=len(capsules)
            capsules.append(dict(name=name if index==0 else f'{name}_part{index:02d}',frame=name,
                start_mm=(center+a@basis).tolist(),end_mm=(center+b@basis).tolist(),radius_mm=float(radius_mm)))
    return capsules,dict(count=len(capsules),nominal_radius_mm=float(radius_mm),mesh_shell_mm=shell_mm,
        box_min=low.tolist(),box_max=high.tolist(),basis=basis.tolist(),center=center.tolist(),
        cross_section_cell_widths_mm=widths.tolist(),cell_corner_distance_mm=float(np.linalg.norm(widths/2)))


def load_link_vertices(urdf_path,name):
    import trimesh
    import yourdfpy
    urdf_path=Path(urdf_path)
    urdf=yourdfpy.URDF.load(urdf_path,load_meshes=False,load_collision_meshes=False)
    clouds=[]
    for shape in urdf.link_map[name].collisions:
        mesh=shape.geometry.mesh
        if mesh is None:raise ValueError('link cover requires collision meshes')
        vertices=np.asarray(trimesh.load(urdf_path.parent/mesh.filename,force='mesh').vertices).copy()
        if mesh.scale is not None:vertices*=np.asarray(mesh.scale)
        origin=shape.origin if shape.origin is not None else np.eye(4)
        clouds.append((vertices@origin[:3,:3].T+origin[:3,3])*1000)
    if not clouds:raise ValueError(f'no collision geometry for {name}')
    return np.concatenate(clouds)
