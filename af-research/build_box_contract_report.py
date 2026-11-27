#!/usr/bin/env python3
"""Bind locally checked source metadata and primary literature to a contract audit."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=BASE/'artifacts/recognition-ranking-repair/box-contract/report.json'
    assert not output.exists()
    root=output.parent
    head=json.loads((root/'head-reference.json').read_text())
    paired_path=BASE/'artifacts/recognition-ranking-repair/paired-face-head-metrics/independent-verification.json'
    paired=json.loads(paired_path.read_text());assert paired['passed']
    names=dict(csv.reader((BASE/'private-data/recognition-v2/raw/classes.csv').open()))
    ids={'person':'/m/01g317','face':'/m/0dzct','head':'/m/04hgtk','body':'/m/02p0tk3','eye':'/m/014sv8'}
    sources={
        'openimages_paper':dict(url='https://arxiv.org/html/1811.00982v2',section='2.4.1',pdf_page=6,
            pdf_url='https://arxiv.org/pdf/1811.00982',local_pdf='sources/openimages-v4-1811.00982.pdf',
            evidence='Author general annotation instructions require tight enclosure of all visible target parts; class-specific edge cases require examples.',
            status='primary_author_method'),
        'openimages_v7':dict(url='https://storage.googleapis.com/openimages/web/factsfigures_v7.html',
            section='Publications; Bounding boxes; Class definitions; Licenses',
            evidence='V7 cites this same method paper for V5-V7, documents target-class exhaustive annotation conditional on positive labels, and licenses annotations CC BY4.0.',status='official_dataset_documentation'),
        'openimages_format':dict(url='https://storage.googleapis.com/openimages/web/download_v7.html',section='Data formats / Bounding boxes',
            evidence='Occluded denotes obstruction by another object; Confidence is a constant placeholder. Different MIDs identify different target classes.',status='official_dataset_documentation'),
        'panasonic_s5_manual':dict(url='https://eww.pavc.panasonic.co.jp/dscoi/DC-S5/P_PP/DC-S5_DVQP2197ZD_full_eng.pdf',
            document_id='DVQP2197ZD',camera='DC-S5',pages=[102,103],pdf_pages_total=598,
            local_pdf='sources/DC-S5_DVQP2197ZD_full_eng.pdf',status='official_original_S5_user_manual',
            evidence='AF display and selected target rules for face/eye/body; head-specific DSP coordinates not described in this section.'),
        'panasonic_s5_release':dict(url='https://na.panasonic.com/news/new-hybrid-full-frame-mirrorless-camera-the-lumix-s5',
            publication_date='2020-09-02',section='3',camera='DC-S5',status='official_manufacturer_function_claim',
            evidence='Original S5 announcement explicitly distinguishes head recognition from eye, face and body, supporting continued tracking when the subject turns.')}
    for row in sources.values():
        if 'local_pdf' in row:row['local_pdf_sha256']=sha(root/row['local_pdf'])
    counts={}
    for split in ('train','validation'):
        eligible=[f for f in head['frames'] if f['split']==split and f['head_annotation_available'] and not f['head_group_or_depiction_present']]
        counts[split]={**head['counts'][split],
            'eligible_frames_excluding_entire_head_flagged_images':len(eligible),
            'eligible_boxes':sum(len(f['boxes']) for f in eligible),
            'excluded_head_flagged_frame_ids':[f['id'] for f in head['frames'] if f['split']==split and f['head_group_or_depiction_present']]}
    facts=[
        dict(id='distinct_classes',status='verified_local_official_metadata',claim='Human face and Human head are separate MIDs, both parts rather than Person subcategories.'),
        dict(id='visible_extent',status='confirmed_published_general_annotation_contract',source='openimages_paper',claim='Open Images targets visible object extent, not amodal completion. Head still means the visible head-class extent.'),
        dict(id='class_specific_face_edges',status='unknown',claim='No publicly verified face-specific exemplar resolves hair, ear, forehead and skin boundaries in this audit.'),
        dict(id='project_mapping',status='verified_local_source',claim='Existing person_face comes solely from Human face. Human head and Human body were not folded into Person; no face-to-head expansion occurred.'),
        dict(id='camera_head_function',status='confirmed_official_manufacturer_claim',source='panasonic_s5_release',claim='Original DC-S5 officially supports head recognition separately from eyes, face and body.'),
        dict(id='manual_ui',status='confirmed_documented_UI',source='panasonic_s5_manual',claim='Yellow selects AF target, white shows alternative detections; eye focus is selected within the yellow area and body can be whole or upper.'),
        dict(id='DSP_head_mapping',status='unknown',claim='No established mapping of layer86/90 or raw model outputs to manufacturer face/head semantics; UI and marketing claims cannot supply that mapping.'),
        dict(id='four_examples',status='verified_saved_metadata_and_geometry',claim='All4 face GTs are Occluded1; corresponding geometrically containing head boxes are Occluded0. Three suppressors remain below headIoU.5; fourth reaches.70358.'),
        dict(id='paired_metrics',status='saved_only_independently_recomputed',claim='OriginalK4 better matches head than face on64 common validation frames, but epoch6 produces182 headFP; changing annotation names does not remove the FP increase.'),
        dict(id='task_drift',status='supported_hypothesis_not_factory_semantics_proof',claim='Training for the project face definition may have shifted an originally more head-like host output; causal factory-label identification remains unproven.')]
    result=dict(status='complete_contract_audit_semantics_unresolved',checked_date='2026-11-27',
        source_sha256={name:sha(BASE/name) for name in ('build_box_contract_report.py','extract_head_reference.py','recognition_data.py','train_recognition.py','verify_face_head_comparison.py')},
        document='docs/S5_FACE_HEAD_CONTRACT.md',document_sha256=sha(BASE/'docs/S5_FACE_HEAD_CONTRACT.md'),
        head_reference_sha256=sha(root/'head-reference.json'),manifest_sha256=head['manifest_sha256'],
        local_class_mapping={k:dict(mid=v,label=names[v]) for k,v in ids.items()},
        hierarchy_contract='Subcategory denotes subclass. Part denotes component. Current person_subclasses recursively includes only Subcategory after finding Person.',
        source_bbox_url='https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
        source_class_url='https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv',
        source_hierarchy_url='https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json',
        sources=sources,claims=facts,head_reference_counts=counts,
        nms_cases=head['four_NMS_cases'],paired_metrics=paired['results'],
        paired_independent_verification_sha256=sha(paired_path),
        annotation_license=dict(owner='Google LLC',license='CC BY4.0',source='openimages_v7'),
        next_verification_contract=[
            'Preserve existing face metrics and immutable labels; report separate face/head matches on identical eligible cohorts, with denominators.',
            'Do not derive instance identity from maximum overlap or assume no head annotation means negative.',
            'Do not select whichever convention has better AP as the original S5 semantic label.',
            'Need camera same-frame outputs, SDK/simulator, or verified firmware dataflow to bind raw output tensors to UI object types and box geometry.',
            'Keep original graph structure and do not resume optimizer until the independently specified localization/ranking checks are met.'
        ],images_read=False,model_inference=False,optimizer_steps=0,test_images_or_predictions_read=False,
        source_labels_changed=False,historical_metrics_changed=False,raw_DSP_semantic_contract_confirmed=False,ready_to_train=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(output);print(sha(output))


if __name__=='__main__':main()
