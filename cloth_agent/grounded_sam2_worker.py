"""Persistent local GPU segmenter. JSONL IO; no camera, robot, or network IO."""
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,Sam2Processor,Sam2Model


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    detector=AutoModelForZeroShotObjectDetection.from_pretrained(root/cfg['detector'],local_files_only=True).to(cfg['device']).eval()
    dp=AutoProcessor.from_pretrained(root/cfg['detector'],local_files_only=True)
    model=Sam2Model.from_pretrained(root/cfg['segmenter'],local_files_only=True).to(cfg['device']).eval()
    sp=Sam2Processor.from_pretrained(root/cfg['segmenter'],local_files_only=True)
    for line in sys.stdin:
        try:
            req=json.loads(line);image=Image.open(req['image']).convert('RGB');full_size=image.size
            crop=req.get('crop') or [0,0,*full_size];image=image.crop(crop)
            start=time.perf_counter();inputs=dp(images=image,text=cfg['prompt'],return_tensors='pt').to(cfg['device'])
            with torch.inference_mode():outputs=detector(**inputs)
            detections=dp.post_process_grounded_object_detection(outputs,inputs.input_ids,
                threshold=cfg['box_threshold'],text_threshold=cfg['text_threshold'],target_sizes=[image.size[::-1]])[0]
            boxes=detections['boxes'].cpu().tolist();scores=detections['scores'].cpu().tolist()
            if not scores:raise ValueError('Grounding DINO found no clothing in current table RGB')
            choice=int(np.argmax(scores));box=boxes[choice]
            inputs=sp(images=image,input_boxes=[[box]],return_tensors='pt').to(cfg['device'])
            with torch.inference_mode():outputs=model(**inputs,multimask_output=True)
            masks=sp.post_process_masks(outputs.pred_masks.cpu(),inputs['original_sizes'].cpu())[0].reshape(-1,image.height,image.width)
            qualities=outputs.iou_scores.detach().cpu().numpy().reshape(-1);best=int(np.argmax(qualities))
            mask=masks[best].numpy().astype(np.uint8)
            if not mask.any():raise ValueError('SAM2 produced empty clothing mask')
            full=np.zeros(full_size[::-1],np.uint8);full[crop[1]:crop[3],crop[0]:crop[2]]=mask
            Image.fromarray(full*255).save(req['output'])
            response=dict(ok=True,boxes=boxes,box_scores=scores,selected_box=choice,selected_mask=best,
                predicted_mask_scores=qualities.tolist(),crop=crop,seconds=time.perf_counter()-start,
                prompt=cfg['prompt'],area_px=int(full.sum()),model='GroundingDINO-tiny + SAM2.1-hiera-small')
        except Exception as exc:response=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
        print(json.dumps(response),flush=True)

if __name__=='__main__':main()
