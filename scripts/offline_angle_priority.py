"""Prepare a fixed-image angle-priority prompt test, without hardware access."""
import argparse,copy,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.offline_pose_guidance_ab import save,run

ANGLE_GUIDANCE='''【展开过程中的朝向纠正】
若衣物尚未充分展开，且衣身主方向相对目标轮廓明显倾斜，下一步优先选择兼顾展开与纠正倾斜的抓放。抓住偏斜一侧的可见布料，沿使衣身转回目标朝向的方向牵拉，而不是继续沿当前倾斜方向拉长；大致摆正后恢复正常展开。这时以纠正倾斜的展开动作为优先，不机械套用对向袖子牵拉。
旋转方向以图片观看方向为准：目标衣身竖直时，当前长轴从左下斜向右上“/”，应逆时针纠正，上端可向左牵拉；从右下斜向左上“\\”，应顺时针纠正，上端可向右牵拉。目标不竖直时按相对目标的倾斜方向判断，不把某个图像方向固定为纠正方向。
这是通过既有抓放牵拉产生的预期衣身转动，不是让夹爪持布旋转，不新增动作类型。无法可靠判断倾斜时正常展开；已充分展开时不为精确对齐继续操作。'''

def prepare(source,out):
 out.mkdir(parents=True,exist_ok=False)
 for name in ['current.png','goal.png','previous.png']:
  if (source/name).exists():shutil.copyfile(source/name,out/name)
 original=json.loads((source/'prepared_A.json').read_text());request=copy.deepcopy(original)
 request['instructions']+='\n'+ANGLE_GUIDANCE
 save(out/'prepared_D.json',request);(out/'prompt_D.txt').write_text(request['instructions'],encoding='utf-8')
 check=copy.deepcopy(request);check['instructions']=original['instructions'];assert check==original
 old=json.loads((source/'protocol.json').read_text())
 save(out/'protocol.json',dict(source=str(source),historical_baseline=['A1','A2'],conditions={'D':'original A + angle-priority prompt; no CV values'},schedule=['D1','D2','D3'],model=request['model'],reasoning=request['reasoning'],images=old['images'],image_sha256=old['image_sha256'],robot_calls=0,retries=0,production_changed=False,limitations=['historical baseline, not contemporaneous randomized control','three repetitions on one image','no physical or full-path validation']))
 print('PREPARED',out,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--source',type=Path,default=Path('reports/offline_pose_guidance_20261108'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a.source,a.output)
 else:run(a.output)
