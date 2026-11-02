"""Official Backend Contract adapter to the operator-owned protected service."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from .control_service import Client
from .supervised import real_tools,validate_request


class RealBackend:
    mutating=frozenset({'move_tcp','move_joints','gripper'})
    def __init__(self,config,frames):
        if config.backend_options['mode']!='real_supervised':raise ValueError('explicit real_supervised mode required')
        self.root=Path(config.root);self.frames=Path(frames);self.client=None;self.closed=False;self.final_status=None
        self.client=Client(config.backend_options['control_socket'],str(uuid.uuid4()))
    def tools(self):return real_tools()
    def validate(self,name,args):
        if self.closed:raise PermissionError('backend closed')
        validate_request(name,args)
    def status(self):
        if self.closed:return self.final_status or {'physical_actions':None,'safety':'UNKNOWN','stop_delivery':'unconfirmed'}
        return self.client.status()
    def call(self,name,args):
        self.validate(name,args)
        value=self.client.call(name,args)
        images=[]
        if name=='observe' and value.get('image_path'):
            source=Path(value['image_path'])
            folder=self.root/'observations'/uuid.uuid4().hex;folder.mkdir(parents=True)
            data=source.read_bytes()
            if hashlib.sha256(data).hexdigest()!=value['sha256']:raise ValueError('live observation image changed')
            (folder/'rgb.png').write_bytes(data)
            (folder/'metadata.json').write_text(json.dumps(value,indent=2))
            target=self.frames/(folder.name+'.png');target.write_bytes(data)
            value={**value,'image_path':'/frames/'+target.name}
            images=[{'type':'image','mimeType':'image/png','data':base64.b64encode(data).decode()}]
        return value,images
    def close(self):
        if self.closed:return
        self.closed=True
        if self.client:self.final_status=self.client.close()
