#!/usr/bin/env python3
"""Private contact sheet from saved validation predictions; never runs a model.

The six frames are selected from IDs and GT sizes only, before reading scores.
Green is target-class GT; orange is saved prediction at score >=0.5.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pretraining_data import need, sha256, validate_box

BASE = Path(__file__).resolve().parent
MODELS = (('picodet', 'person', 'PicoDet | PERSON'),
          ('blazeface', 'person_face', 'BlazeFace | FACE | NATIVE'),
          ('blazeface_s5_size', 'person_face', 'BlazeFace | FACE | 384 x 256'))
SCORE = .5
GREEN = '#28d58b'
ORANGE = '#ffa143'


def small_faces(frame):
    return [box for box in frame['boxes'] if box['class'] == 'person_face' and
            math.sqrt((box['bbox_xyxy'][2]-box['bbox_xyxy'][0]) *
                      (box['bbox_xyxy'][3]-box['bbox_xyxy'][1]) *
                      384*256/(frame['width']*frame['height'])) < 32]


def choose_frames(frames):
    ordered = sorted((frame for frame in frames if frame['split'] == 'validation'), key=lambda f: f['id'])
    need(len(ordered) >= 6, 'At least six validation frames are required')
    selected = [(frame, 'first_three_sorted_validation_ids') for frame in ordered[:3]]
    used = {frame['id'] for frame, _reason in selected}
    extra = [frame for frame in ordered if frame['id'] not in used and small_faces(frame)][:3]
    need(len(extra) == 3, 'Need three additional validation frames containing reference-small faces')
    return selected + [(frame, 'first_three_remaining_sorted_ids_with_reference_small_face') for frame in extra]


def font(size):
    for name in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                 '/System/Library/Fonts/Helvetica.ttc',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, ValueError):
            pass
    return ImageFont.load_default(size=size)


def read_saved_run(runs_root, model, target_class, manifest_hash, validation):
    folder = runs_root/model
    path, report_path = folder/'predictions/real.json', folder/'report.json'
    need(path.is_file() and report_path.is_file(), 'All three completed runs are required: '+model)
    report = json.loads(report_path.read_text())
    expected = report['saved_artifacts']['predictions/real.json']
    need(sha256(path) == expected['sha256'] and path.stat().st_size == expected['bytes'], 'Saved prediction identity differs: '+model)
    prediction = json.loads(path.read_text())
    need(report['model'] == prediction['model'] == model, 'Model/profile mismatch')
    need(report['dataset_manifest_sha256'] == prediction['dataset_manifest_sha256'] == manifest_hash, 'Dataset mismatch')
    need(prediction['control'] == 'real', 'Contact sheet only reads real predictions')
    by_id = {row['id']: row for row in prediction['frames']}
    need(len(by_id) == len(prediction['frames']) and set(by_id) == set(validation), 'Validation coverage differs')
    for fid, row in by_id.items():
        for detection in row['detections']:
            need(detection['class'] == target_class, 'Wrong target class in saved run')
            need(isinstance(detection['score'], (int, float)) and math.isfinite(detection['score']) and
                 0 <= detection['score'] <= 1, 'Invalid saved score')
            validate_box(detection['bbox_xyxy'], validation[fid]['width'], validation[fid]['height'])
    transform = report['model_metadata']['input_transform']
    if model == 'blazeface':
        need(transform.get('resize_policy') == 'native', 'Native profile is not declared native')
    elif model == 'blazeface_s5_size':
        need(transform.get('width') == 384 and transform.get('height') == 256 and
             transform.get('keep_ratio') is False, 'S5-size profile geometry differs')
    return by_id, {'model': model, 'target_class': target_class, 'input_transform': transform,
                   'model_id': report['model_metadata']['model_id'],
                   'prediction_file': path.relative_to(runs_root).as_posix(), 'prediction_sha256': sha256(path),
                   'report_file': report_path.relative_to(runs_root).as_posix(), 'report_sha256': sha256(report_path)}


def fit_text(draw, text, x, y, maximum, size=17, color='#eeeeee'):
    selected = font(size)
    if draw.textlength(text, font=selected) > maximum:
        while text and draw.textlength(text+'...', font=selected) > maximum:
            text = text[:-1]
        text += '...'
    draw.text((x, y), text, fill=color, font=selected)


def draw_boxes(draw, boxes, offset, scale, color, line_width):
    x, y = offset
    sx, sy = scale
    for box in boxes:
        coordinates = [x+box[0]*sx, y+box[1]*sy, x+box[2]*sx, y+box[3]*sy]
        draw.rectangle(coordinates, outline=color, width=line_width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--runs-root', type=Path, default=BASE/'artifacts/external-baselines/runs')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/external-baselines/preview.png')
    parser.add_argument('--width', type=int, default=1600)
    args = parser.parse_args()
    need(args.width >= 1200, 'Contact sheet width must be at least1200 pixels for three readable columns')
    need(not args.output.resolve().is_relative_to(args.runs_root.resolve()), 'Do not write into saved model runs')
    manifest_hash = sha256(args.dataset)
    dataset = json.loads(args.dataset.read_text())
    need(dataset.get('task') == 'recognition_detection', 'Unexpected dataset task')
    validation = {frame['id']: frame for frame in dataset['frames'] if frame['split'] == 'validation'}
    selected = choose_frames(dataset['frames'])
    # The deterministic IDs are fixed before any prediction files are opened.
    runs, run_metadata = {}, []
    for model, target, _title in MODELS:
        runs[model], metadata = read_saved_run(args.runs_root, model, target, manifest_hash, validation)
        run_metadata.append(metadata)
    provenance = dataset['provenance']
    provenance_path = (args.dataset.parent/provenance['path']).resolve()
    need(provenance_path.is_relative_to(args.dataset.parent.resolve()) and
         sha256(provenance_path) == provenance['sha256'], 'Provenance identity differs')
    margin, gap = 18, 12
    column_width = (args.width-2*margin-2*gap)//3
    image_height, row_height, header = 340, 408, 138
    canvas = Image.new('RGB', (args.width, header+len(selected)*row_height+54), '#10151d')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), 'External detector comparison - saved validation predictions', fill='white', font=font(27))
    draw.text((margin, 55), 'GREEN: ground truth', fill=GREEN, font=font(18))
    draw.text((265, 55), 'ORANGE: saved score >= 0.50', fill=ORANGE, font=font(18))
    draw.text((600, 55), 'Fixed selection | no new inference | private diagnostic', fill='#c0c8d5', font=font(18))
    for col, (model, _target, title) in enumerate(MODELS):
        left = margin+col*(column_width+gap)
        fit_text(draw, title, left+8, 93, column_width-16, 20)
        if model == 'picodet':
            transform = run_metadata[col]['input_transform']
            subtitle = f"Input {transform.get('width')} x {transform.get('height')} | person boxes only"
        elif model == 'blazeface':
            subtitle = 'Original image resolution | face boxes only'
        else:
            subtitle = 'Stretched 384 x 256, mapped back | face only'
        fit_text(draw, subtitle, left+8, 118, column_width-16, 15, '#adb9ca')
    selection_rows = []
    for row_index, (frame, reason) in enumerate(selected):
        path = (args.dataset.parent/frame['path']).resolve()
        need(path.is_relative_to(args.dataset.parent.resolve()) and sha256(path) == frame['sha256'], 'Image content changed')
        with Image.open(path) as im:
            need(im.size == (frame['width'], frame['height']), 'Image dimensions changed')
            source = im.convert('RGB')  # Preserve stored-pixel orientation used by labels.
        scale = min((column_width-16)/frame['width'], image_height/frame['height'])
        display_size = (max(1, round(frame['width']*scale)), max(1, round(frame['height']*scale)))
        display_scale = (display_size[0]/frame['width'], display_size[1]/frame['height'])
        tile = source.resize(display_size, Image.Resampling.LANCZOS)
        display_counts = {}
        top = header+row_index*row_height
        for col, (model, target, _title) in enumerate(MODELS):
            left = margin+col*(column_width+gap)
            draw.rounded_rectangle((left, top+8, left+column_width, top+row_height-4), radius=7, fill='#1c2430')
            gt = [box['bbox_xyxy'] for box in frame['boxes'] if box['class'] == target]
            predicted = [d for d in runs[model][frame['id']]['detections'] if d['class'] == target and d['score'] >= SCORE]
            caption = f"{row_index+1}. GT {len(gt)} | Pred {len(predicted)} | original {frame['width']} x {frame['height']}"
            fit_text(draw, caption, left+8, top+14, column_width-16, 17)
            offset = (left+(column_width-display_size[0])//2, top+42+(image_height-display_size[1])//2)
            canvas.paste(tile, offset)
            # Exact original-image xyxy coordinates; no learned or regenerated imagery.
            draw_boxes(draw, gt, offset, display_scale, GREEN, 4)
            draw_boxes(draw, [d['bbox_xyxy'] for d in predicted], offset, display_scale, ORANGE, 2)
            footer = frame['source_image_id']
            if small_faces(frame):
                footer += f' | S5-ref small faces: {len(small_faces(frame))}'
            fit_text(draw, footer, left+8, top+row_height-25, column_width-16, 15, '#b9c5d5')
            display_counts[model] = {'target_class': target, 'GT_boxes': len(gt), 'prediction_boxes_score_ge_0_5': len(predicted)}
        selection_rows.append({
            'row': row_index+1, 'id': frame['id'], 'selection_reason': reason,
            'image_path_relative_to_dataset': frame['path'], 'image_sha256': frame['sha256'],
            'width': frame['width'], 'height': frame['height'], 'small_face_count_s5_reference': len(small_faces(frame)),
            'source_dataset': frame['source_dataset'], 'source_image_id': frame['source_image_id'],
            'source_official_split': frame.get('source_official_split'), 'source_url': frame.get('source_url'),
            'download_url': frame.get('download_url'), 'license_url': frame.get('license_url'),
            'source_author': frame.get('source_author'), 'source_author_profile': frame.get('source_author_profile'),
            'source_title': frame.get('source_title'), 'display_counts': display_counts})
    draw.text((margin, canvas.height-42), 'Private diagnostic. Native and 384-input runs have different costs. Six selected frames are not an accuracy estimate.',
              fill='#c0c8d5', font=font(17))
    need(sha256(args.dataset) == manifest_hash, 'Manifest changed while drawing')
    for item in run_metadata:
        need(sha256(args.runs_root/item['prediction_file']) == item['prediction_sha256'] and
             sha256(args.runs_root/item['report_file']) == item['report_sha256'], 'Saved run changed while drawing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, format='PNG')
    selection = {'privacy': 'Private derived-photo diagnostic; keep in ignored artifacts, do not commit or publish with code.',
        'selection_rule': 'First3 lexically sorted validation IDs, then first3 remaining sorted validation IDs containing at least one face with sqrt(area after384x256 reference stretch)<32.',
        'selection_uses_predictions': False, 'selected_frames': selection_rows,
        'dataset_manifest_sha256': manifest_hash, 'provenance_sha256': provenance['sha256'],
        'runs': run_metadata, 'prediction_display_threshold': SCORE, 'ground_truth_color': GREEN,
        'prediction_color': ORANGE, 'box_coordinates': 'Original-image continuous xyxy mapped to the actual rounded display dimensions of an aspect-preserving tile.',
        'image_transform': 'Stored RGB orientation, display resize and annotated boxes; source photos unchanged.',
        'script_sha256': sha256(Path(__file__)), 'preview_sha256': sha256(args.output),
        'preview_dimensions': list(canvas.size), 'new_model_inference': False, 'optimizer_steps': 0,
        'test_images_read': False, 'limits': ['Deterministic examples, not aggregate performance or cherry-picked successes.',
            'PicoDet shows person GT/predictions; BlazeFace columns show face GT/predictions.',
            'Native-resolution and384x256 face profiles have different computational costs.',
            'Small-face detection gains and whole-face AP can differ; consult complete saved metrics.']}
    selection_path = args.output.parent/'selection.json'
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'preview': str(args.output), 'selection': str(selection_path),
                      'dimensions': list(canvas.size), 'selected_ids': [frame['id'] for frame, _reason in selected],
                      'new_model_inference': False, 'test_images_read': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
