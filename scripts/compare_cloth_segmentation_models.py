"""Offline matched-box SAM1/SAM2 comparison. No camera or robot calls."""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
from PIL import Image
import cv2,torch


def save(root,name,method,rgb,mask,metrics):
 out=root/name/method;out.mkdir(exist_ok=True)
 Image.fromarray(mask.astype('uint8')*255).save(out/'mask.png')
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 overlay=np.array(rgb).copy();cv2.drawContours(overlay,contours,-1,(0,255,70),2)
 Image.fromarray(overlay).save(out/'overlay.png')
 metrics['area_px']=int(mask.sum());(out/'result.json').write_text(json.dumps(metrics,indent=2))
 print(name,method,metrics,flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='reports/segmentation_model_comparison_20261108');p.add_argument('--method',choices=['sam2','sam1'],required=True);a=p.parse_args()
 root=Path(a.root);manifest=json.loads((root/'manifest.json').read_text());device='cuda';torch.set_num_threads(4)
 if a.method=='sam2':
  from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection,Sam2Processor,Sam2Model
  from huggingface_hub import snapshot_download
  cache='/tmp/cloth-seg-models'
  dp=snapshot_download('IDEA-Research/grounding-dino-tiny',cache_dir=cache,local_files_only=True)
  sp=snapshot_download('facebook/sam2.1-hiera-small',cache_dir=cache,local_files_only=True)
  processor=AutoProcessor.from_pretrained(dp);detector=AutoModelForZeroShotObjectDetection.from_pretrained(dp).to(device).eval()
  segmenter=Sam2Model.from_pretrained(sp).to(device).eval();segment_processor=Sam2Processor.from_pretrained(sp)
 else:
  from segment_anything import sam_model_registry,SamPredictor
  predictor=SamPredictor(sam_model_registry['vit_h'](checkpoint='/home/CNS2026330003/agentAuto/agent/checkpoints/sam/sam_vit_h_4b8939.pth').to(device).eval())
 for e in manifest['images']:
  name=e['name'];im=Image.open(e['image']).convert('RGB');rgb=np.array(im)
  if a.method=='sam2':
   start=time.perf_counter();inp=processor(images=im,text=manifest['prompt'],return_tensors='pt').to(device)
   with torch.inference_mode():outputs=detector(**inp)
   det=processor.post_process_grounded_object_detection(outputs,inp.input_ids,threshold=manifest['box_threshold'],text_threshold=manifest['text_threshold'],target_sizes=[im.size[::-1]])[0]
   torch.cuda.synchronize();seconds=time.perf_counter()-start
   data=dict(boxes=det['boxes'].cpu().tolist(),scores=det['scores'].cpu().tolist(),labels=det.get('text_labels',det.get('labels')),seconds=seconds,prompt=manifest['prompt'])
   if len(data['scores']):data['selected_index']=int(np.argmax(data['scores']))
   (root/name/'detection.json').write_text(json.dumps(data,indent=2))
  else:data=json.loads((root/name/'detection.json').read_text())
  if not data['scores']:
   save(root,name,a.method,rgb,np.zeros(rgb.shape[:2],bool),dict(status='NO_DETECTION'));continue
  box=data['boxes'][data['selected_index']];start=time.perf_counter()
  if a.method=='sam2':
   inp=segment_processor(images=im,input_boxes=[[box]],return_tensors='pt').to(device)
   with torch.inference_mode():outputs=segmenter(**inp,multimask_output=True)
   masks=segment_processor.post_process_masks(outputs.pred_masks.cpu(),inp['original_sizes'].cpu())[0].reshape(-1,im.height,im.width)
   scores=outputs.iou_scores.detach().cpu().numpy().reshape(-1);best=int(np.argmax(scores));mask=masks[best].numpy().astype(bool)
  else:
   with torch.inference_mode():
    predictor.set_image(rgb);masks,scores,_=predictor.predict(box=np.array(box),multimask_output=True)
   best=int(np.argmax(scores));mask=masks[best]
  torch.cuda.synchronize();elapsed=time.perf_counter()-start
  save(root,name,a.method,rgb,mask,dict(status='OK',box=box,segmentation_seconds=elapsed,detection_seconds=data['seconds'],predicted_mask_scores=list(map(float,scores)),selected_mask=best,manual_prompts=False))

if __name__=='__main__':main()
