"""Generated offline policy execution and interface validation, no robot imports."""
import ast
import numpy as np
import jsonschema
from PIL import Image, ImageDraw
from contracts import action_schema

def checked_compile(code):
    tree=ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n,(ast.ImportFrom,ast.With,ast.AsyncWith,ast.ClassDef,ast.Global,ast.Nonlocal,ast.While,ast.AsyncFunctionDef)):
            raise ValueError('Unsupported generated syntax '+type(n).__name__)
        if isinstance(n,ast.Import) and any(a.name not in ['numpy','math','cv2'] for a in n.names):
            raise ValueError('Nonallowlisted import')
        if isinstance(n,ast.Name) and (n.id.startswith('__') or n.id in ['open','eval','exec','compile','getattr','setattr','globals','locals','input','breakpoint']):
            raise ValueError('Forbidden generated name')
        if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or any(w in n.attr.lower() for w in ['load','save','file','ctypes','dump','memmap','imread','imwrite','capture','writer'])):
            raise ValueError('Forbidden generated attribute')
    f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_action']
    assert len(f)==1 and [a.arg for a in f[0].args.args]==['rgb','target_corner']
    return compile(tree,'candidate_skill.py','exec')

def invoke(code,prompt,goal,rgb_path,request):
    rgb=np.asarray(Image.open(rgb_path).convert('RGB'));calls=[]
    def vision(image,text):
        assert not calls and np.array_equal(image,rgb)
        calls.append(True)
        return request(text,rgb.shape[1],rgb.shape[0])
    env={'VISION_MODEL':vision,'FROZEN_PROMPT':prompt,'LOCKED_GOAL':goal}
    exec(checked_compile(code),env)
    action=env['generate_action'](rgb,goal[7:-1])
    jsonschema.validate(action,action_schema(goal))
    errors=[]
    if action['decision']=='ACTION':
        if action['target_state'] not in ['OCCLUDED','SUSPECTED_OCCLUDED']:errors.append('Target not supported as hidden')
        if action['grasp_px']==action['target_px']:errors.append('Zero displacement')
        if not all(0<=p[0]<rgb.shape[1] and 0<=p[1]<rgb.shape[0] for p in [action['grasp_px'],action['target_px']]):errors.append('Out of image')
    if not action['experimental_only']:errors.append('Not experimental-only')
    return action,errors

def overlay(image,action,dest):
    import math
    im=Image.open(image).convert('RGB');d=ImageDraw.Draw(im)
    if action['decision']=='ACTION':
        a,b=action['grasp_px'],action['target_px']
        d.line([tuple(a),tuple(b)],fill='yellow',width=5)
        ang=math.atan2(b[1]-a[1],b[0]-a[0])
        d.polygon([tuple(b),(b[0]-14*math.cos(ang-.45),b[1]-14*math.sin(ang-.45)),(b[0]-14*math.cos(ang+.45),b[1]-14*math.sin(ang+.45))],fill='yellow')
        for p,c,name in [(a,'lime','Pa'),(b,'red','Pb')]:
            d.ellipse((p[0]-6,p[1]-6,p[0]+6,p[1]+6),fill=c);d.text((p[0]+8,p[1]),name,fill=c)
    im.save(dest)
