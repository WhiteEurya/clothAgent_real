"""Read-only Viser inspection of captured dual-camera clouds; no hardware access."""
import argparse,json,time
from pathlib import Path
import numpy as np
import viser


def read_cloud(path):
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line: raise ValueError('Invalid PLY header')
            if line.strip()==b'end_header':break
        data=np.fromfile(f,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('r','u1'),('g','u1'),('b','u1')])
    return np.stack([data[k] for k in 'xyz'],1),np.stack([data[k] for k in 'rgb'],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--parameters',type=Path,required=True);p.add_argument('--port',type=int,default=8091);a=p.parse_args()
    meta=json.loads((a.capture/'report.json').read_text());cal=json.loads(Path(meta['calibration']).read_text());params=json.loads(a.parameters.read_text())
    clouds={}
    for name,key in [('zed','world_from_zed_mm'),('realsense','world_from_realsense_color_mm')]:
        raw,color=read_cloud(a.capture/f'{name}.ply');t=np.array(cal[key])
        if name=='realsense':t=t@np.array(meta['realsense_depth_to_color_mm'])
        scale=float(params['cameras'][name]['scale']);native=(raw-t[:3,3])@t[:3,:3];corrected=(native*scale)@t[:3,:3].T+t[:3,3]
        clouds[name]=(raw/1000,corrected/1000,color)
    server=viser.ViserServer(host='127.0.0.1',port=a.port,label='双相机点云对齐检查（只读）')
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('第三个板位置的静态采集。橙色 ZED、蓝色 RealSense。\n\n深度修正使用既有参数，未重新拟合；仍有残差。未补洞、未平滑，不连接相机或机械臂。')
    mode=server.gui.add_dropdown('点云版本',('原始点云','深度比例修正'),initial_value='深度比例修正')
    coloring=server.gui.add_dropdown('颜色',('相机区分色','真实 RGB'),initial_value='相机区分色')
    visible={name:server.gui.add_checkbox('显示 '+name.upper(),initial_value=True) for name in clouds}
    size=server.gui.add_slider('点大小（mm）',min=.5,max=5.,step=.5,initial_value=1.5)
    top=server.gui.add_button('俯视');side=server.gui.add_button('侧视');oblique=server.gui.add_button('斜视')
    handles={name:server.scene.add_point_cloud('/'+name,points=values[1].astype(np.float32),colors=(255,145,60) if name=='zed' else (50,160,255),point_size=.0015,point_shape='circle',precision='float32') for name,values in clouds.items()}
    server.scene.add_frame('/world',axes_length=.15,axes_radius=.002)
    center=np.median(np.concatenate([x[1] for x in clouds.values()]),axis=0)
    def camera(client,kind='oblique'):
        client.camera.look_at=tuple(center)
        client.camera.up=(0,1,0) if kind=='top' else (0,0,1)
        delta={'top':(0,0,1.5),'side':(0,-1.5,.08),'oblique':(1,-1,1)}[kind]
        client.camera.position=tuple(center+delta)
    @server.on_client_connect
    def connect(client):camera(client)
    def update(_):
        with server.atomic():
            for name,(raw,corrected,rgb) in clouds.items():
                h=handles[name];h.points=(raw if mode.value=='原始点云' else corrected).astype(np.float32)
                h.colors=rgb if coloring.value=='真实 RGB' else ((255,145,60) if name=='zed' else (50,160,255))
                h.visible=visible[name].value;h.point_size=size.value/1000
    for control in [mode,coloring,size,*visible.values()]:control.on_update(update)
    for button,kind in [(top,'top'),(side,'side'),(oblique,'oblique')]:
        def handler(event,kind=kind):
            if event.client is not None:camera(event.client,kind)
        button.on_click(handler)
    print(f'READY http://127.0.0.1:{server.get_port()} | {a.capture}',flush=True)
    while True:time.sleep(1)

if __name__=='__main__':main()
