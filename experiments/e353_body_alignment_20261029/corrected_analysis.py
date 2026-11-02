from pathlib import Path
import sys,json
ROOT=Path('/mnt/newssd/sja/clothAgent_real');sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'));import engine
R=ROOT/'reports/e353_body_alignment_20261029';engine.OUT=R
f=R/'calls/jpeg_independent_structure_analysis';raw=(f/'input.txt').read_text();instruction,tail=raw.split('\n\n',1);payload=json.loads(tail.rsplit('\n图片顺序',1)[0]);payload['alignment']['limitations'].append('Contours are automatically thresholded; inspect original RGB for fine edge identity. Chest anchors define local body frame, not whole-body rigid truth.')
r=engine.call('independent_corrected_contours',instruction,payload,[R/n for n in ['before.png','after.png','aligned_overlay.png','aligned_contours_corrected.png','body_anchor_fit.png']],engine.read(f/'schema.json'),'independent_e353_analysis',['E353'])
engine.save(R/'analysis_corrected.json',engine.require(r))
engine.save(R/'analysis_selection.json',{'final_call':'jpeg_independent_corrected_contours','first_call_retained':True,'reason':'First automatic contour merged background pixels; segmentation brightness threshold/morphology corrected. Fresh independent request; no first model output supplied.','chat_history_included':False})
print(json.dumps(r['output'],ensure_ascii=False,indent=2))
