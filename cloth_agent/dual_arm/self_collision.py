"""Same-arm solid geometry checks using the existing URDF/FCL mesh loader.

All input joints and clearances use degrees/mm. FCL uses metres internally.
Only shapes on the same rigid link and adjacent arm links are excluded.
"""
import threading
import numpy as np
from .collision.scene import collision_mesh
from .geometry import DualArmError


class SelfCollisionChecker:
    def __init__(self, model, margin_mm, clearance_mm):
        import fcl
        self.fcl, self.model = fcl, model
        self.required = 2*margin_mm+clearance_mm
        self.lock = threading.RLock()
        self.shapes=[]
        parents={j.child:j for j in model.urdf.robot.joints}
        links=['link_base']+[f'link{i}' for i in range(1,model.config.axis+1)]
        def add(name,frame,host,obj,local,radius):
            weights=np.zeros(model.config.axis);ancestors=set();link=frame
            while link!='link_base':
                joint=parents[link]
                if joint.name in model.names:
                    index=model.names.index(joint.name);weights[index]=radius;ancestors.add(index)
                elif joint.type!='fixed':raise DualArmError('unsupported self-collision attachment joint')
                if joint.origin is not None:radius+=np.linalg.norm(joint.origin[:3,3])*1000
                link=joint.parent
            self.shapes.append(dict(name=name,frame=frame,host=host,obj=obj,local=local,
                                    weights=weights,ancestors=ancestors))
        for link in links:
            collisions=model.urdf.link_map[link].collisions
            if not collisions:raise DualArmError(f'{link}: missing self-collision geometry')
            for i,c in enumerate(collisions):
                mesh=collision_mesh(c.geometry,model.config.urdf.parent)
                local=np.eye(4) if c.origin is None else c.origin
                vertices=np.asarray(mesh.vertices)
                faces=np.column_stack((np.full(len(mesh.faces),3),mesh.faces)).flatten()
                obj=fcl.CollisionObject(fcl.Convex(vertices,len(mesh.faces),faces))
                radius=np.linalg.norm(vertices@local[:3,:3].T+local[:3,3],axis=1).max()*1000
                add(f'{link}/{i}',link,link,obj,local,radius)
        # Tool/camera envelopes remain in the existing inter-arm/table checker.
        # Those coarse assembly envelopes overlap neighbouring wrist links even
        # at Home; they are not suitable same-arm link collision geometry.
        neighbors={frozenset((j.parent,j.child)) for j in model.urdf.robot.joints}
        self.pairs=[]
        for i,a in enumerate(self.shapes):
            for b in self.shapes[i+1:]:
                if a['host']==b['host']:continue
                if a['frame'] in links and b['frame'] in links and frozenset((a['host'],b['host'])) in neighbors:continue
                # Common ancestor rotations preserve relative shape distance.
                weights=a['weights']+b['weights']
                for index in a['ancestors'] & b['ancestors']:weights[index]=0
                self.pairs.append((a,b,weights))

    def check_box(self, center_deg, half_deg):
        with self.model._lock, self.lock:
            self.model.update(center_deg)
            for s in self.shapes:
                frame=self.model.frame(s['frame']);frame[:3,3]/=1000
                frame=frame@s['local']
                s['obj'].setTransform(self.fcl.Transform(frame[:3,:3],frame[:3,3]))
            for a,b,weights in self.pairs:
                result=self.fcl.CollisionResult()
                self.fcl.collide(a['obj'],b['obj'],self.fcl.CollisionRequest(),result)
                distance=0. if result.is_collision else self.fcl.distance(
                    a['obj'],b['obj'],self.fcl.DistanceRequest(),self.fcl.DistanceResult())*1000
                bound=float(weights@np.radians(np.abs(half_deg)))
                if not np.isfinite(distance) or distance<=self.required+bound:
                    raise DualArmError(f"{self.model.config.arm_id}: self-collision clearance: "
                        f"{a['name']} vs {b['name']}, distance={distance:.3f} mm, "
                        f"required={self.required+bound:.3f} mm (sweep={bound:.3f})")
