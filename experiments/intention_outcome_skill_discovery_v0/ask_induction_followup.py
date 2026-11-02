"""One isolated API follow-up; replay prior induction only, no coordinator advice."""
from pathlib import Path
import json, hashlib
import sys
from transport import call, save, read, sha

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'reports/intention_outcome_skill_discovery_v0_20261030/calls/induction/attempt_0'
OUT=REPO/'reports/intention_outcome_induction_followup_20261030'

def main():
    OUT.mkdir(exist_ok=True)
    question='我已经知道某次下摆牵拉意外改善了袖口可见性。现在，如果我希望主动实现袖口显露，应该怎样从一张新的 RGB 中决定操作位置和方向？我想看看如果我们问已经得到刚才结论的 agent 下一步怎么做，它会怎么看。'
    transcript=[{'role':'user','content':(SOURCE/'input.txt').read_text()},
                {'role':'assistant','content':(SOURCE/'output.json').read_text()}]
    manifest=read(SOURCE/'manifest.json')
    images=[Path(x['source']) for x in manifest['images']]
    for p,x in zip(images,manifest['images']): assert sha(p)==x['sha256']
    save(OUT/'provenance.json',{
        'mode':'reconstructed prior induction context; not resumed persistent session',
        'reason':'original induction used --ephemeral',
        'source':str(SOURCE),'source_input_sha256':sha(SOURCE/'input.txt'),
        'source_output_sha256':sha(SOURCE/'output.json'),'question':question,
        'coordinator_conversation_included':False,'coordinator_method_suggestion_included':False,
        'later_synthesis_or_test_results_included':False,'new_rgb_provided':False,
        'images':manifest['images'],'max_new_model_calls':1,'robot_access':False})
    name='followup_permitted' if '--permitted-retry' in sys.argv else 'followup'
    result=call(OUT,name,
        '以下是一次先前独立分析的完整用户输入和模型回答，按时间顺序提供。原会话是临时会话；现在用这些记录重建上下文进行一次续问，并非声称恢复了原会话。附件图片与先前输入一致，顺序不变。请基于这些记录直接回答最后的用户问题，用中文。没有提供新的当前RGB。',
        {'prior_transcript':transcript,'new_user_question':question},images,
        {'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False})
    if result['status']!='COMPLETE':
        print(json.dumps(result,ensure_ascii=False));return
    (OUT/'answer.md').write_text(result['output']['answer'])
    print(json.dumps({'status':result['status'],'latency_s':result['latency_s'],'usage':result['usage'],'answer':str(OUT/'answer.md')},ensure_ascii=False))

if __name__=='__main__':main()
