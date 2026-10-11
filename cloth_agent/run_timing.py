"""Monotonic per-round timing, including failed and interrupted rounds."""
from contextlib import contextmanager
import json
from pathlib import Path
import time


class RoundTiming:
    def __init__(self,folder,index):
        self.folder=Path(folder);self.index=index;self.seconds={};self.status='RUNNING'
    def __enter__(self):
        self.started=time.perf_counter();return self
    @contextmanager
    def measure(self,name):
        start=time.perf_counter()
        try:yield
        finally:self.seconds[name]=self.seconds.get(name,0.)+time.perf_counter()-start
    def __exit__(self,kind,error,tb):
        self.seconds['round_total_s']=time.perf_counter()-self.started
        record=dict(status=self.status if error is None else kind.__name__,seconds=self.seconds)
        if error is not None:record['error']=str(error)
        (self.folder/'timing.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        labels={'observation_s':'准备与采集','gpt_planning_s':'GPT 规划','gpt_code_repair_s':'GPT 代码修正','trajectory_planning_s':'轨迹规划验证',
                'preview_s':'预览生成','preview_wait_s':'等待点击','execution_s':'运动执行','round_total_s':'本轮总计'}
        print(f'Round {self.index} 耗时：'+'；'.join(f'{label} {self.seconds[key]:.2f}s' for key,label in labels.items() if key in self.seconds),flush=True)
