#!/usr/bin/env python3
"""Run restored original S5 weights on one local photo with the K4 layout.

Host research inference only. Does not train, modify firmware, or claim camera
parity. Small-face misses remain a measured limitation.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from extract import MODELS, require, write_json
from preprocessing_investigation import prepare, back_to_photo
from pretraining_data import sha256
from train_recognition import Detector, MODEL, detections, state_digest


def run(args):
    require(0 <= args.score <= 1, 'Score threshold must be in [0,1]')
    require(not args.output.exists(), 'Use a new output folder')
    require(sha256(args.firmware)==MODELS[MODEL], 'Expected unmodified 44_hm_d_nw_sng.bin')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    with Image.open(args.image) as image:
        rgb=image.convert('RGB')
    # Stored-pixel coordinates, identical to the dataset evaluation contract.
    tensor,transform=prepare(rgb,dict(color='pillow_l',mean=117.,scale=1.,geometry=args.geometry))
    detector=Detector(args.firmware.read_bytes(),'gemm_k4')
    detector.model.eval().requires_grad_(False)
    before=state_digest(detector.model.state_dict())
    try:
        with torch.inference_mode():
            logits,delta=detector(tensor)
            items=back_to_photo(detections(logits,delta,detector.anchors,{'width':384,'height':256}),transform)
        require(state_digest(detector.model.state_dict())==before,'Model changed')
    finally:
        detector.close()
    args.output.mkdir(parents=True)
    write_json(args.output/'predictions.json',{
        'model':MODEL,'source_model_sha256':MODELS[MODEL],'model_state_sha256':before,
        'input_sha256':sha256(args.image),'width':rgb.width,'height':rgb.height,
        'layout':'gemm_k4','geometry':args.geometry,'input_transform':transform,
        'detections':items,'preview_score_threshold':args.score,'optimizer_steps':0,
        'camera_numerical_parity':False,'small_face_fix_verified':False,
        'limitations':['Single-image host hypothesis, not real-camera inference.',
                       'Measured small-face misses remain; do not interpret a drawn box as a verified match.',
                       'No EXIF rotation; boxes refer to stored image pixels.']})
    preview=rgb.copy();draw=ImageDraw.Draw(preview)
    size=max(12,round(min(rgb.size)/40))
    try: font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',size)
    except OSError: font=ImageFont.load_default(size=size)
    count=0
    for item in items:
        if item['score']<args.score:continue
        color='#00df91' if item['class']=='person' else '#ffaf32'
        box=item['bbox_xyxy'];draw.rectangle(box,outline=color,width=max(2,size//8))
        draw.text((box[0]+2,max(0,box[1]-size-3)),f"{item['class']} {item['score']:.2f}",font=font,fill=color,stroke_width=1,stroke_fill='black')
        count+=1
    preview.save(args.output/'preview.png')
    print(json.dumps({'output':str(args.output),'preview_detections':count,'optimizer_steps':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firmware',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry',choices=('stretch','letterbox'),default='stretch')
    p.add_argument('--score',type=float,default=.5,help='Preview only; saved JSON retains all decoder outputs')
    run(p.parse_args())
