#!/usr/bin/env python3
"""Private v2 transferred-head comparison from saved validation predictions only.

Prepare the deterministic selection before epoch6 is available. Rendering never
loads a model, searches for a best epoch, evaluates test data, or changes sources.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from pretraining_data import need, sha256
from visualize_external_baselines import choose_frames, fit_text, font, read_saved_run, small_faces

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
GREEN, ORANGE = '#28d58b', '#ffa143'
SCORE = .5
COLUMNS = (
    ('baseline_original', 'S5 K4 | ORIGINAL ANCHORS', 'Original weights | scales 3,6,9,16,32'),
    ('baseline_small', 'S5 K4 | TRANSFER INIT | EPOCH 0', 'Analytic head transfer | scales 1,2,4,8,16'),
    ('epoch-06', 'S5 K4 | TRANSFER INIT | EPOCH 6', 'Fixed final epoch | initialized by analytic transfer'),
    ('blazeface_s5_size', 'BLAZEFACE 384 | REFERENCE', 'External face-only reference | S5 deployment unverified'),
)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def prepare(args):
    need(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected validation dataset')
    data = json.loads(args.dataset.read_text())
    selected = choose_frames(data['frames'])
    validation = {f['id']: f for f in data['frames'] if f['split'] == 'validation'}
    need(len(validation) == 80 and len(selected) == 6, 'Expected validation80 and fixed six')
    train_plan = json.loads((args.run/'plan.json').read_text())
    need(train_plan['manifest_sha256'] == MANIFEST_SHA and train_plan['layout'] == 'gemm_k4' and
         train_plan['epochs'] == 6 and train_plan['test_inference'] is False, 'Unexpected training plan')
    need(set(train_plan['validation_ids']) == set(validation), 'Training plan validation split differs')
    need(train_plan['initialization'] == 'analytic_log_nearest_anchor_transfer', 'Expected transferred-head v2 initialization')
    transfer = train_plan['transfer_provenance']
    need(transfer['input_state_sha256'] == train_plan['original_state_sha256'] and
         transfer['transferred_state_sha256'] == train_plan['initial_state_sha256'] and
         transfer['old_scale_index_by_new_scale'] == [0, 0, 0, 2, 3], 'Transfer state or fixed mapping differs')
    previous_path = BASE/'artifacts/recognition-small-anchor-v1-preview/selection.json'
    previous = json.loads(previous_path.read_text())
    need([r['id'] for r in previous['selected_frames']] == [f['id'] for f, _ in selected], 'V2 must use the same six as v1')
    need(not args.output.resolve().is_relative_to(args.run.resolve()), 'Keep visualization outside frozen training run')
    provenance = data['provenance']
    provenance_path = (args.dataset.parent/provenance['path']).resolve()
    need(provenance_path.is_relative_to(args.dataset.parent.resolve()) and
         sha256(provenance_path) == provenance['sha256'], 'Dataset provenance changed')
    sources = {name: sha256(BASE/name) for name in
               ('visualize_small_anchor_training_v2.py', 'visualize_external_baselines.py', 'pretraining_data.py')}
    selection = dict(selection_rule='Exact visualize_external_baselines.choose_frames: first3 sorted validation IDs, '
        'then first3 remaining sorted validation IDs with reference-small face. No prediction-dependent selection.',
        selection_uses_predictions=False, selected_frames=[dict(id=f['id'], reason=reason,
            image_sha256=f['sha256'], width=f['width'], height=f['height'],
            small_face_count_s5_reference=len(small_faces(f))) for f, reason in selected],
        manifest_sha256=MANIFEST_SHA, provenance_sha256=provenance['sha256'],
        training_plan_sha256=sha256(args.run/'plan.json'), source_sha256=sources,
        previous_v1_selection_sha256=sha256(previous_path),
        initialization=train_plan['initialization'], original_state_sha256=train_plan['original_state_sha256'],
        transferred_epoch0_state_sha256=train_plan['initial_state_sha256'],
        transfer_old_scale_indices=[0, 0, 0, 2, 3],
        transfer_equivalence_limit='Only mapped slots preserve unclipped boxes before exponent clamp; duplicate slots and omitted old scales mean the full original proposal set is not preserved.',
        fixed_epoch=6, columns=[dict(id=key, title=title, subtitle=subtitle) for key, title, subtitle in COLUMNS],
        score_threshold=SCORE, gt_color=GREEN, prediction_color=ORANGE,
        class_labels={'P': 'person', 'F': 'person_face'}, model_inference=False,
        test_images_read=False, fixed_validation_image_reads_after_predictions_ready=6,
        privacy='Private derived-photo diagnostic in ignored artifacts; do not publish photos with code.')
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output/'selection-plan.json'
    if selection_path.exists():
        need(json.loads(selection_path.read_text()) == selection, 'Frozen selection/source plan changed')
    else:
        save_json(selection_path, selection)
    return data, validation, selected, train_plan, selection


def read_training_predictions(args, key, training_plan, validation):
    folder = args.run/key
    prediction_path, metrics_path = folder/'predictions.json', folder/'metrics.json'
    need(prediction_path.is_file() and metrics_path.is_file(), 'Saved epoch/baseline predictions not ready: '+key)
    checkpoint_path = folder/'checkpoint.pt'
    if key != 'baseline_original':
        need(checkpoint_path.is_file(), 'Checkpoint save has not completed: '+key)
    document = json.loads(prediction_path.read_text())
    expected_anchor = training_plan['original_anchor_sha256'] if key == 'baseline_original' else training_plan['small_anchor_sha256']
    need(document['dataset_manifest_sha256'] == MANIFEST_SHA and
         document['plan_sha256'] == sha256(args.run/'plan.json') and
         document['anchor_sha256'] == expected_anchor, 'Saved prediction plan/anchor identity differs')
    if key != 'epoch-06':
        expected_state = training_plan['original_state_sha256'] if key == 'baseline_original' else training_plan['initial_state_sha256']
        need(document['model_state_sha256'] == expected_state, 'Original/transfer epoch0 state differs')
    rows = {r['id']: r for r in document['frames']}
    need(len(rows) == len(document['frames']) and set(rows) == set(validation), 'Saved predictions are not exactly validation80')
    for fid, row in rows.items():
        frame = validation[fid]
        for detection in row['detections']:
            need(detection['class'] in ('person', 'person_face'), 'Unexpected predicted class')
            score, box = detection['score'], detection['bbox_xyxy']
            need(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1, 'Invalid confidence')
            need(len(box) == 4 and all(math.isfinite(v) for v in box) and
                 -1e-6 <= box[0] < box[2] <= frame['width']+1e-6 and
                 -1e-6 <= box[1] < box[3] <= frame['height']+1e-6, 'Invalid original-image box')
    metadata = dict(id=key, prediction_file=(Path(key)/'predictions.json').as_posix(),
        prediction_sha256=sha256(prediction_path), metrics_file=(Path(key)/'metrics.json').as_posix(),
        metrics_sha256=sha256(metrics_path), model_state_sha256=document['model_state_sha256'],
        anchor_sha256=document['anchor_sha256'], plan_sha256=document['plan_sha256'],
        input_policy=training_plan['preprocessing'], source_model_sha256=training_plan['model_sha256'])
    if checkpoint_path.is_file():
        metadata.update(checkpoint_file=(Path(key)/'checkpoint.pt').as_posix(),
                        checkpoint_file_sha256=sha256(checkpoint_path),
                        checkpoint_scope='File hash only; checkpoint tensors are never loaded by the visualization.')
    return rows, metadata


def draw_labelled_box(draw, box, offset, scale, color, label, width, label_inside=False):
    ox, oy = offset; sx, sy = scale
    x1, y1, x2, y2 = [ox+box[0]*sx, oy+box[1]*sy, ox+box[2]*sx, oy+box[3]*sy]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    label_font = font(13)
    text_box = draw.textbbox((0, 0), label, font=label_font)
    tw, th = text_box[2]-text_box[0]+6, text_box[3]-text_box[1]+5
    # GT labels above; prediction labels inside/below the top edge to separate
    # their provenance even when the rectangle overlaps exactly.
    ty = y1+2 if label_inside else max(oy, y1-th)
    draw.rectangle((x1, ty, x1+tw, ty+th), fill='#10151d')
    draw.text((x1+3, ty+1-text_box[1]), label, font=label_font, fill=color)


def render_canvas(selected, images, runs, row_start=0):
    cw, rh, header = 510, 410, 154
    canvas = Image.new('RGB', (4*cw, header+rh*len(selected)+48), '#10151d')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), 'Transferred-head training v2 | fixed six validation examples | saved predictions', fill='white', font=font(27))
    draw.text((16, 51), 'GREEN = GT     ORANGE = prediction score >= 0.50     P = person     F = face',
              fill='#e1e7ef', font=font(19))
    draw.text((16, 79), 'A displayed box does not imply an IoU >= 0.50 match. Six examples are not an accuracy estimate.',
              fill='#bac6d8', font=font(18))
    for col, (_key, title, subtitle) in enumerate(COLUMNS):
        fit_text(draw, title, col*cw+12, 112, cw-24, 19)
        fit_text(draw, subtitle, col*cw+12, 137, cw-24, 14, '#adbacb')
    display_counts = []
    for row, (frame, reason) in enumerate(selected):
        original = images[frame['id']]
        scale = min((cw-24)/original.width, (rh-84)/original.height)
        size = (max(1, round(original.width*scale)), max(1, round(original.height*scale)))
        tile = original.resize(size, Image.Resampling.LANCZOS)
        row_counts = {}
        for col, (key, _title, _subtitle) in enumerate(COLUMNS):
            left, top = col*cw, header+row*rh
            draw.rounded_rectangle((left+6, top+7, left+cw-6, top+rh-4), radius=6, fill='#1c2430')
            allowed = ('person_face',) if key == 'blazeface_s5_size' else ('person', 'person_face')
            gt = [b for b in frame['boxes'] if b['class'] in allowed]
            pred = [d for d in runs[key][frame['id']]['detections'] if d['class'] in allowed and d['score'] >= SCORE]
            counts = {cls: dict(gt=sum(b['class'] == cls for b in gt), predictions=sum(d['class'] == cls for d in pred))
                      for cls in allowed}
            caption = ' | '.join(f"{'P' if cls == 'person' else 'F'} GT{counts[cls]['gt']}/Pred{counts[cls]['predictions']}" for cls in allowed)
            fit_text(draw, f'{row_start+row+1}. {caption}', left+12, top+14, cw-24, 16)
            offset = (left+(cw-size[0])//2, top+42+((rh-84)-size[1])//2)
            canvas.paste(tile, offset)
            display_scale = (size[0]/frame['width'], size[1]/frame['height'])
            for item in gt:
                label = 'GT P' if item['class'] == 'person' else 'GT F'
                draw_labelled_box(draw, item['bbox_xyxy'], offset, display_scale, GREEN, label, 3)
            for item in pred:
                label = ('P' if item['class'] == 'person' else 'F')+f" {item['score']:.2f}"
                draw_labelled_box(draw, item['bbox_xyxy'], offset, display_scale, ORANGE, label, 2, True)
            subtitle = frame['source_image_id']+f' | reference-small faces {len(small_faces(frame))}'
            fit_text(draw, subtitle, left+12, top+rh-29, cw-24, 14, '#b4c1d3')
            row_counts[key] = counts
        display_counts.append(dict(id=frame['id'], reason=reason, counts=row_counts))
    draw.text((16, canvas.height-34), 'Private diagnostic | stored-image coordinates | no new inference | no test data | BlazeFace is face-only',
              fill='#b5c2d3', font=font(17))
    return canvas, display_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--external-runs', type=Path, default=BASE/'artifacts/external-baselines/runs')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2-preview-label-fix')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    data, validation, selected, training_plan, selection = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(status='selection_and_sources_frozen', selected_ids=[f['id'] for f, _ in selected],
                              epoch6_ready=(args.run/'epoch-06/checkpoint.pt').is_file(),
                              images_read=False, predictions_read=False), indent=2))
        return
    runs, identities = {}, []
    for key, _title, _subtitle in COLUMNS[:3]:
        runs[key], metadata = read_training_predictions(args, key, training_plan, validation)
        identities.append(metadata)
    runs['blazeface_s5_size'], external = read_saved_run(args.external_runs, 'blazeface_s5_size', 'person_face', MANIFEST_SHA, validation)
    identities.append(external)
    images, photo_sources = {}, []
    for frame, reason in selected:
        path = (args.dataset.parent/frame['path']).resolve()
        need(path.is_relative_to(args.dataset.parent.resolve()) and sha256(path) == frame['sha256'], 'Selected photo hash differs')
        with Image.open(path) as image:
            need(image.size == (frame['width'], frame['height']), 'Stored dimensions differ')
            images[frame['id']] = image.convert('RGB')
        photo_sources.append(dict(id=frame['id'], reason=reason, image_sha256=frame['sha256'],
            image_path_relative_to_dataset=frame['path'], source_image_id=frame['source_image_id'],
            source_dataset=frame['source_dataset'], source_url=frame.get('source_url'),
            download_url=frame.get('download_url'), license_url=frame.get('license_url'),
            source_author=frame.get('source_author'), source_author_profile=frame.get('source_author_profile'),
            source_title=frame.get('source_title')))
    canvas, counts = render_canvas(selected, images, runs)
    canvas.save(args.output/'comparison.png')
    outputs = {'comparison.png': dict(sha256=sha256(args.output/'comparison.png'), dimensions=list(canvas.size))}
    for page in range(3):
        sheet, _ = render_canvas(selected[page*2:page*2+2], images, runs, page*2)
        name = f'comparison-page-{page+1}.png'; sheet.save(args.output/name)
        outputs[name] = dict(sha256=sha256(args.output/name), dimensions=list(sheet.size))
    need(sha256(args.dataset) == MANIFEST_SHA and sha256(args.run/'plan.json') == selection['training_plan_sha256'], 'Bound plans changed')
    need(all(sha256(BASE/name) == value for name, value in selection['source_sha256'].items()), 'Visualization sources changed')
    for item in identities[:3]:
        for field, hash_field in [('prediction_file', 'prediction_sha256'), ('metrics_file', 'metrics_sha256'),
                                 ('checkpoint_file', 'checkpoint_file_sha256')]:
            if field in item:
                need(sha256(args.run/item[field]) == item[hash_field], 'Saved training artifact changed while drawing')
    need(sha256(args.external_runs/external['prediction_file']) == external['prediction_sha256'] and
         sha256(args.external_runs/external['report_file']) == external['report_sha256'], 'External reference changed')
    save_json(args.output/'selection.json', dict(**selection,
        selection_plan_sha256=sha256(args.output/'selection-plan.json'), photo_sources=photo_sources,
        saved_prediction_sources=identities, displayed_counts=counts, output_images=outputs,
        image_transform='Stored RGB orientation, aspect-preserving display resize; boxes map with actual rounded display dimensions.',
        limits=['Only saved score>=.5 boxes are displayed; box presence is not an IoU match.',
                'Same fixed six selected validation examples as the prior external comparison.',
                'New-anchor epoch0 uses analytic head transfer; it is not unchanged original head values.',
                'Epoch6 is fixed by the training plan, never replaced by an earlier better epoch.',
                'Transfer duplicates selected old slots; this does not reproduce the complete old proposal set or create automatic small-face ability.',
                'BlazeFace is an external face-only reference; its success does not establish S5 deployability.'],
        new_optimizer_steps=0, checkpoint_tensors_loaded=False))
    print(json.dumps(dict(status='rendered', output=str(args.output), outputs=outputs,
                          new_model_inference=False, test_images_read=False), indent=2))


if __name__ == '__main__':
    main()
