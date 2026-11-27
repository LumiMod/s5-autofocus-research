#!/usr/bin/env python3
"""Private saved-prediction comparison, using the prior fixed six-frame rule."""
import gzip
import json
from pathlib import Path
from PIL import Image, ImageDraw
from pretraining_data import sha256
from visualize_external_baselines import choose_frames, draw_boxes, font

BASE=Path(__file__).resolve().parent


def main():
    manifest=BASE/'private-data/recognition-v2/manifest.json'
    dataset=json.loads(manifest.read_text())
    root=manifest.parent
    selected=choose_frames(dataset['frames'])
    folder=BASE/'artifacts/recognition-diagnosis-v3/k4-validation80'
    sources=[]
    path=folder/'k4_stretch/predictions.json.gz'
    doc=json.loads(gzip.decompress(path.read_bytes()))['real']
    sources.append(('S5 K4 | PERSON + FACE',doc,path))
    for model,title in [('picodet','PicoDet | PERSON'),('blazeface_s5_size','BlazeFace 384 | FACE'),('blazeface','BlazeFace native | FACE')]:
        path=BASE/'artifacts/external-baselines/runs'/model/'predictions/real.json'
        sources.append((title,json.loads(path.read_text()),path))
    rows=[]
    for title,doc,path in sources:
        assert doc['dataset_manifest_sha256']==sha256(manifest)
        rows.append((title,{r['id']:r['detections'] for r in doc['frames']}))
    cw,rh=500,380
    canvas=Image.new('RGB',(cw*len(rows),100+rh*len(selected)),'#15191f')
    draw=ImageDraw.Draw(canvas)
    draw.text((14,12),'S5 K4 / external references | validation | untrained',font=font(25),fill='white')
    draw.text((14,48),'Green: ground truth. Orange: saved predictions >= 0.5. Boxes alone do not imply IoU >= 0.5.',font=font(19),fill='white')
    identities=[]
    for row,(frame,reason) in enumerate(selected):
        image_path=root/frame['path']; assert sha256(image_path)==frame['sha256']
        with Image.open(image_path) as im: original=im.convert('RGB')
        scale=min((cw-20)/original.width,(rh-76)/original.height)
        size=(round(original.width*scale),round(original.height*scale))
        resized=original.resize(size,Image.Resampling.LANCZOS)
        for col,(title,predictions) in enumerate(rows):
            x,y=col*cw+10,100+row*rh
            draw.text((x,y),title,font=font(18),fill='white')
            draw.text((x,y+24),frame['id'][-32:],font=font(14),fill='#bac4d0')
            offset=(x+(cw-20-size[0])//2,y+48)
            canvas.paste(resized,offset)
            classes={'person','person_face'} if col==0 else {'person'} if col==1 else {'person_face'}
            for item in frame['boxes']:
                if item['class'] in classes: draw_boxes(draw,[item['bbox_xyxy']],offset,(size[0]/original.width,size[1]/original.height),'#28d58b',2)
            for item in predictions[frame['id']]:
                if item['score']>=.5 and item['class'] in classes:
                    draw_boxes(draw,[item['bbox_xyxy']],offset,(size[0]/original.width,size[1]/original.height),'#ffa143',2)
        identities.append({'id':frame['id'],'reason':reason,'sha256':frame['sha256']})
    out=folder/'comparison.png';canvas.save(out)
    (folder/'preview-selection.json').write_text(json.dumps({'manifest_sha256':sha256(manifest),
        'frames':identities,'sources':[{'title':t,'sha256':sha256(p),'path':str(p.relative_to(BASE))} for t,d,p in sources],
        'image_sha256':sha256(out),'score':.5,'training':False},ensure_ascii=False,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
