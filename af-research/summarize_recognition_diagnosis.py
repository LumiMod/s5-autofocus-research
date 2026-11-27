#!/usr/bin/env python3
"""Summarize existing diagnosis JSON and verify file identities; never infer/train.

This standard-library-only program does not import model code or open firmware,
photographs, dataset manifests, or test predictions. Hashing the training probe
files listed by an existing verification index is not numerical revalidation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
ART = 'artifacts/recognition-diagnosis/'
OUTPUT = BASE / 'evidence/2026-11-27/recognition-diagnosis/summary.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def select(document, keys):
    return {key: document[key] for key in keys if key in document}


def metric_rows(metrics):
    return {mode: {'mean_ap50': value['mean_ap'],
                   'person_ap50': value['classes']['person']['ap_at_iou'],
                   'face_ap50': value['classes']['person_face']['ap_at_iou']}
            for mode, value in metrics.items()}


def require_private_path_free(value):
    # Reject accidental personal paths even in copied source prose.
    encoded = json.dumps(value, ensure_ascii=False)
    if re.search(r'/Users/|/home/|/private/|file://|[A-Za-z]:\\\\Users\\\\', encoded):
        raise ValueError('Personal absolute path found in summary')


class Evidence:
    def __init__(self):
        self.observed, self.files, self.bindings, self.unbound = {}, {}, [], []

    def bytes(self, relative, publish=True):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Evidence reference must be repository-relative')
        path = BASE / relative
        data = path.read_bytes()
        name, checksum = relative.as_posix(), digest(data)
        if name in self.observed and self.observed[name] != checksum:
            raise RuntimeError('Source changed during summary: ' + name)
        self.observed[name] = checksum
        if publish:
            self.files[name] = {'sha256': checksum, 'bytes': len(data)}
        return data

    def read(self, relative):
        return json.loads(self.bytes(relative))

    def source_binding(self, origin, values, snapshots=None):
        if not values:
            self.unbound.append({'source': origin,
                                 'status': 'No recorded script hash; not independently verified'})
            return
        matches, missing, mismatches, historical = [], [], [], []
        snapshots = snapshots or {}
        for name, expected in sorted(values.items()):
            if not (BASE / name).is_file():
                missing.append(name)
                continue
            actual = digest(self.bytes(name))
            if actual == expected:
                matches.append(name)
            else:
                mismatches.append({'file': name, 'recorded_sha256': expected,
                                   'current_sha256': actual})
            snapshot = snapshots.get(name)
            if snapshot and (BASE / snapshot).is_file():
                observed = digest(self.bytes(snapshot))
                historical.append({'file': name, 'snapshot': snapshot,
                    'snapshot_sha256': observed, 'matches_recorded_hash': observed == expected})
        recovered = {row['file'] for row in historical if row['matches_recorded_hash']}
        self.bindings.append({'source': origin, 'recorded_hashes': values,
                              'matching_current_files': matches, 'missing_files': missing,
                              'mismatches': mismatches, 'historical_snapshots': historical,
                              'all_recorded_hashes_match_current': not missing and not mismatches,
                              'all_recorded_hashes_match_current_or_snapshot':
                                  all(name in matches or name in recovered for name in values)})

    def verify_index(self, relative, index):
        folder = Path(relative).parent
        checked, mismatches, missing = 0, [], []
        for name, record in sorted(index.items()):
            path = folder / name
            if not (BASE / path).is_file():
                missing.append(path.as_posix())
                continue
            data = self.bytes(path, publish=False)
            checked += 1
            if digest(data) != record['sha256'] or len(data) != record['bytes']:
                mismatches.append(path.as_posix())
        return {'index_source': relative, 'listed_files': len(index),
                'files_hashed': checked, 'missing': missing, 'mismatches': mismatches,
                'all_listed_file_identities_match': not missing and not mismatches,
                'numerical_metrics_recomputed_by_summarizer': False}

    def stable(self):
        for name, expected in self.observed.items():
            if digest((BASE / name).read_bytes()) != expected:
                raise RuntimeError('Source changed during summary; rerun: ' + name)


def preprocessing(e):
    review = e.read(ART + 'preprocessing/review-summary.json')
    reports = {row['source']: e.read(row['source']) for row in review['rows']}
    for source, report in reports.items():
        e.source_binding(source, report.get('source_hashes_end'))
        if report['status'] != 'complete' or report['source_hashes_start'] != report['source_hashes_end']:
            raise ValueError('Incomplete or changed preprocessing run: ' + source)
        if report['split'] not in ('train', 'validation'):
            raise ValueError('Unexpected preprocessing split')
    rows = []
    for row in review['rows']:
        report = reports[row['source']]
        observed = report['metrics'][row['variant']]
        if row['map50'] != {mode: value['mean_ap'] for mode, value in observed['metrics'].items()}:
            raise ValueError('Preprocessing review disagrees with its underlying report')
        if row['stage'] == 'validation':
            rows.append({'source': row['source'], 'layout': row['layout'],
                         'variant': row['variant'], 'images': len(report['frame_ids']),
                         'metrics': metric_rows(observed['metrics']),
                         'real_minus_strongest_control_map50': observed['real_minus_strongest_control_map50']})
    folder = ART + 'preprocessing/o4_i4_hw__direct/development/'
    response = {}
    for variant in ('baseline_gray117', 'gray117_times_input_scale'):
        report = e.read(folder + variant + '/report.json')
        response[variant] = [select(layer, ('index', 'name', 'signal_over_real_rms',
            'real_max_abs', 'int16_range_hypothesis_limit',
            'real_fraction_beyond_int16_range_hypothesis'))
            for layer in report['layers_after_candidate_relu'] if layer['index'] in (1, 74, 90)]
    first = next(iter(reports.values()))
    return {'outcome': review['outcome'], 'development_images_per_run': 32,
            'predeclared_preprocessing_variants_per_layout': 6,
            'independent_self_checks_reported': review['independent_self_checks'],
            'validation': rows, 'layer_response_train32': response,
            'input_evidence': select(first['input_evidence'],
                ('embedded_array', 'input_node_output_scale_candidate', 'embedded_value_sign', 'unverified')),
            'binding_limit': 'Run source hashes and summary agreement checked; state hash and metrics are reported evidence, not independently recomputed inference.'}


def packing(e):
    source = ART + 'packing/findings.json'
    findings = e.read(source)
    if findings['scope']['optimizer_steps'] != 0 or findings['scope']['validation_or_test_pixels_read']:
        raise ValueError('Packing scope differs from the frozen diagnostic round')
    e.source_binding(source, findings.get('source_hashes'))
    semantic = e.read(ART + 'packing/probe-semantic-checks.json')
    e.source_binding(ART + 'packing/probe-semantic-checks.json',
                     {'weight_layout_investigation.py': semantic['script_sha256']})
    candidates = []
    for stage in ('initial', 'followup'):
        path = ART + 'packing/' + stage + '/runtime-summary.json'
        report = e.read(path)
        e.source_binding(path, {'weight_layout_investigation.py': report['plan']['script_sha256']},
                         {'weight_layout_investigation.py': ART + 'packing/' + stage + '/weight_layout_investigation-source.py'})
        for row in report['candidates']:
            if row['candidate']['name'] in ('legacy_o4_i4_hw', 'conv1_HOW_o4_i4_hw', 'transpose_all_legacy'):
                candidates.append({'source': path, 'candidate': row['candidate'],
                                   'metrics': metric_rows(row['metrics'])})
    return {'status': findings['status'], 'scope': findings['scope'],
            'findings': findings['findings'], 'selected_diagnostic_examples': candidates,
            'semantic_checks_reported_passed': semantic['all_passed'],
            'semantic_check_count': len(semantic['checks']),
            'binding_limit': 'Current scripts match recorded identities where indicated; semantic checks and numerical predictions were not rerun.'}


def anchors(e):
    verification_path = ART + 'anchors/verification.json'
    verification = e.read(verification_path)
    identities = e.verify_index(verification_path, verification['files'])
    report = e.read(ART + 'anchors/report.json')
    if report['optimizer_steps'] != 0 or report['test_inference_performed'] or report['validation_inference_performed']:
        raise ValueError('Anchor inference scope differs from the frozen diagnostic round')
    plan = e.read(ART + 'anchors/predeclared-plan.json')
    e.source_binding(ART + 'anchors/report.json', report.get('source_hashes'))
    e.source_binding(ART + 'anchors/predeclared-plan.json', plan.get('sources'))
    geometry = e.read(ART + 'anchors/geometry.json')
    static = e.read(ART + 'anchors/static-evidence.json')
    proposals = {}
    for model, record in static.items():
        table = record['proposal_records'][0]['remaining_512_bytes']
        proposals[model] = {
            **select(record, ('sha256', 'proposal_numeric_fields_identical', 'parameter_blocks_identical',
                'both_proposals_share_bbox_layer', 'shared_bbox_layer', 'smallest_square_anchor_side',
                'bbox_output_scale', 'bbox_signed_i16_direct_range_conditional', 'delta_scaling_evidence')),
            'proposal_count': len(record['proposal_records']),
            'ratio_candidates': record['proposal_records'][0]['parameter_block']['ratio_candidates'],
            'scale_candidates': record['proposal_records'][0]['parameter_block']['scale_candidates'],
            'exp_like_table': select(table, ('candidate_function', 'candidate_domain', 'interval_width',
                'slope_exact_word_matches', 'slope_word_count', 'positive_intercept_exact_word_matches',
                'positive_intercept_word_count', 'empirical_all_intercept_word_matches',
                'first_slope_saturation_index', 'range_warning', 'unknowns')),
            'complete_anchor_signature_matches_in_bounded_search': sum(
                row['count'] for row in record['candidate_anchor_table_signature_search'])}
    return {'status': report['status'], 'inference_training_images': report['fixed_train_samples'],
            'decoder_candidates_per_layout': len(plan['candidates']), 'layouts': plan['layouts'],
            'verification_reported': {k: v for k, v in verification.items() if k != 'files'},
            'verification_file_identity_audit': identities,
            'geometry_only': [{'design': row['design'], 'split': row['split'],
                'scales': row['scales'], 'anchor_shapes_per_location': 25,
                'change_from_stored_scales': ('none' if row['design'] == 'stored_scales' else
                    'Replace all five scales, retaining five ratios and 25 anchor shapes; not append one small anchor.'),
                'summary': row['summary']} for row in geometry['results']],
            'geometry_warning': geometry['warning'], 'static_proposal_evidence': proposals,
            'observations': report['observations'],
            'binding_limit': 'Verification index files were rehashed; its recomputed-prediction, firmware-word and frozen-state checks are reported results, not rerun here.'}


def outer_axes(e):
    report = e.read(ART + 'outer-axes/report.json')
    if report['training_steps'] != 0 or report['test_access'] or report['photograph_inference']:
        raise ValueError('Outer-axis investigation is no longer static-only')
    verification = e.read(ART + 'outer-axes/verification.json')
    plan = e.read(ART + 'outer-axes/predeclared-plan.json')
    e.source_binding(ART + 'outer-axes/report.json', report.get('source_hashes'))
    e.source_binding(ART + 'outer-axes/predeclared-plan.json', plan.get('sources'))
    actual = e.files[ART + 'outer-axes/report.json']['sha256']
    return {**select(report, ('status', 'axis_orders', 'score_equivalence_classes',
                'cross_model_macro_rank_spearman', 'lowest_macro_score_class', 'frozen_candidate_count',
                'qualifying_score_classes', 'photograph_inference', 'training_steps', 'test_access', 'limitations')),
            'verification_reported': verification,
            'verification_report_hash_matches_current': actual == verification['report_sha256'],
            'binding_limit': 'Static metric verification counts are source-reported; this summary did not decode or permute weights.'}


def arithmetic(e):
    source = ART + 'arithmetic/runtime-summary.json'
    report = e.read(source)
    plan = e.read(ART + 'arithmetic/arithmetic-plan.json')
    frozen = e.read(ART + 'arithmetic/frozen-probe-plan.json')
    if frozen != report['plan']:
        raise ValueError('Arithmetic runtime plan differs from frozen plan')
    if not frozen['no_training_no_validation_no_test']:
        raise ValueError('Arithmetic scope differs from the frozen diagnostic round')
    e.source_binding(ART + 'arithmetic/arithmetic-plan.json', {
        'arithmetic_investigation.py': plan['arithmetic_script_sha256'],
        'weight_layout_investigation.py': plan['packing_script_sha256']})
    e.source_binding(source, {'weight_layout_investigation.py': report['plan']['script_sha256']})
    return {'scope': plan['scope'], 'images': len(frozen['frames']),
            'independent_self_checks_reported': plan['independent_checks_passed'],
            'candidates': [{'candidate': row['candidate']['name'], 'metrics': metric_rows(row['metrics'])}
                           for row in report['candidates']],
            'limitations': plan['limitations'] + [
                'Detector reads post-quantization logits/deltas, then detections recomputes floating-point softmax; graph softmax-output quantization does not drive AP.',
                'Unsigned mode applies only at nodes marked ReLU; other operators still use signed bounds. Neither mode is recovered original-camera arithmetic.'],
            'binding_limit': 'Plan binds only two scripts; no full dependency or before/after model-state hashes recorded for this arithmetic run. Those missing bindings were not independently verified.'}


def arm(e):
    source = ART + 'arm-input-boundary.json'
    report = e.read(source)
    e.source_binding(source, {'arm_input_boundary.py': report['script_sha256']})
    historical = {**select(report, ('source_file', 'source_sha256', 'load_base', 'component_bytes',
                 'direct_mapping_end_exclusive', 'capstone_version', 'scope', 'answers', 'bounded_negative_findings')),
              'observations': [select(row, ('id', 'addresses', 'fact', 'limit')) for row in report['observations']],
              'binding_limit': 'Report and script identities checked; original firmware component and disassembled blocks not independently reread/redecoded by this summary.'}
    output = {'historical_chain': historical, 'current_boundary': report['answers'],
              'historical_unresolved_slot_limitation_superseded': False}
    # Later scatter evidence supersedes the old unresolved RAM-table limit.
    # Preserve report identities and addresses, never its private QEMU path.
    additions = []
    for path in sorted((BASE / ART).glob('arm*scatter*.json')):
        relative = path.relative_to(BASE).as_posix()
        item = e.read(relative)
        if path.name == 'arm-input-scatter-boundary.json':
            binding = dict(item.get('dependency_sha256', {}))
            binding['arm_input_scatter_boundary.py'] = item['script_sha256']
            e.source_binding(relative, binding)
            mapping = item['table_mapping']
            old = item.get('historical_emulation_corroboration', {})
            additions.append({'source': relative,
                **select(item, ('source_file', 'source_sha256', 'supersedes_limitations_of', 'scope', 'answer', 'limitations')),
                'table_mapping': select(mapping, ('destination_ram_va', 'formula', 'original_source_va',
                    'program_file_offset', 'whole_copy_source_sha256', 'slot_entries')),
                'historical_emulation_corroboration': {
                    **select(old, ('report_sha256', 'recorded_copy_hash_matches_original_source', 'evidence_type')),
                    'report_basename': Path(old.get('path', 'unrecorded')).name,
                    'historical_report_independently_read_by_summarizer': False},
                'observations': [select(row, ('id', 'addresses', 'fact', 'limit', 'evidence'))
                                 for row in item.get('observations', [])]})
            output['current_boundary'] = item['answer']
            output['historical_unresolved_slot_limitation_superseded'] = True
        else:
            e.unbound.append({'source': relative,
                             'status': 'Optional scatter report schema not independently source-bound by this summary'})
            additions.append({'source': relative, 'reported': select(item, (
                'status', 'scope', 'conclusion', 'answers', 'limitations', 'source_sha256')),
                'reported_observations': [select(row, ('id', 'addresses', 'fact', 'limit'))
                                          for row in item.get('observations', [])]})
    output['optional_scatter_reports'] = additions
    producer_path = ART + 'arm-input-direct-callers.json'
    if (BASE / producer_path).is_file():
        producer = e.read(producer_path)
        e.source_binding(producer_path, {'arm_input_direct_xrefs.py': producer['script_sha256']})
        output['descriptor_producer_boundary'] = {
            'source': producer_path,
            **select(producer, ('source_file', 'source_sha256', 'source_bytes', 'targets',
                'thumb_branch_candidate_count', 'arm_branch_candidate_count', 'conclusion')),
            'direct_branch_hits': [select(row, ('va', 'target', 'isa', 'mnemonic'))
                                   for row in producer['direct_branch_hits']],
            'exact_pointer_word_match_counts': {key: len(value)
                for key, value in producer['exact_pointer_word_occurrences'].items()},
            'caller_block_identity': select(producer['caller_block'],
                ('name', 'start_va', 'end_va_exclusive', 'file_offset', 'bytes', 'sha256')),
            'binding_limit': 'Report/script identities checked; bounded negative call-search findings were not independently redisassembled here.'}
    postboot_path = ART + 'arm-input-postboot-callers.json'
    if (BASE / postboot_path).is_file():
        postboot = e.read(postboot_path)
        e.source_binding(postboot_path, {'arm_input_postboot_xrefs.py': postboot['script_sha256']})
        output['bounded_other_component_caller_search'] = {
            'source': postboot_path,
            **select(postboot, ('program_sha256', 'targets', 'scope', 'conclusion')),
            'components': [{**select(row, ('file', 'source_sha256', 'bytes', 'load_base', 'load_table_va',
                'storage_source', 'thumb_candidates', 'arm_candidates', 'branch_scan_skipped', 'reason')),
                'direct_branch_hits': [select(hit, ('va', 'target', 'isa', 'mnemonic'))
                                       for hit in row.get('direct_branch_hits', [])],
                'exact_pointer_word_match_counts': {key: len(value)
                    for key, value in row.get('pointer_word_offsets', {}).items()}}
                for row in postboot['components']],
            'binding_limit': 'Only the script hash is recorded; dependency script hashes are unavailable. Firmware hashes and bounded search results were not independently recomputed here. Font resource branch scanning was explicitly skipped.'}
    return output


def self_check():
    try:
        require_private_path_free({'bad': '/Users/example/private-data'})
    except ValueError:
        pass
    else:
        raise AssertionError('Absolute-path guard failed')
    require_private_path_free({'source': 'artifacts/recognition-diagnosis/report.json'})
    assert select({'x': 2, 'secret': 3}, ('x', 'missing')) == {'x': 2}
    assert digest(b'abc') == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    checks = self_check()
    if args.self_check:
        print('Summary self-checks passed:', checks)
        return
    evidence = Evidence()
    evidence.bytes(Path(__file__).name)
    result = {'schema_version': 1, 'ready_to_resume_recognition_training': False,
        'verified_recognition_fix': False,
        'evidence_kind': 'Derived index and file-identity audit of existing reports; not original numerical-run proof.',
        'summary_execution': {'optimizer_steps': 0, 'model_inference': False,
            'test_inference': False, 'firmware_or_model_weights_loaded': False,
            'photographs_opened': False, 'dataset_manifests_opened': False,
            'independent_summary_self_checks': checks},
        'diagnostic_round_scope_reported': {'optimizer_steps': 0, 'test_inference': False,
            'validation_inference': 'Preprocessing only: frozen candidates on 80 validation images.',
            'training_subsets': 'Preprocessing/packing/arithmetic use first32 sorted train IDs; anchor decoder uses first8. Outer-axis/ARM probes are static.'},
        'metric_definition': 'Two-class all-points AP at IoU0.5, averaged across person/face; not COCO mAP.',
        'preprocessing': preprocessing(evidence), 'packing': packing(evidence),
        'anchors': anchors(evidence), 'outer_axes': outer_axes(evidence),
        'arithmetic': arithmetic(evidence), 'arm_input_boundary': arm(evidence),
        'unresolved_before_training': [
            'No preprocessing/layout/arithmetic candidate has established original-model image-dependent recognition.',
            'Input-lane permutations, inter-layer channel correspondence and upsample routing/phase addressing remain unresolved.',
            'Camera pixel format, ROI/crop/resize, subtraction direction and input scaling remain unresolved.',
            'Proposal exp range reduction, argument scaling, coefficient signedness and overflow remain unresolved.',
            'Original DSP numerical equivalence has not been measured; AP changes cannot substitute for it.',
            'Smaller anchors improve counterfactual geometry only; they cannot be substituted into original trained heads without evidence.']}
    result['file_identities_at_summary_time'] = dict(sorted(evidence.files.items()))
    result['source_binding_checks'] = evidence.bindings
    result['unavailable_source_bindings'] = evidence.unbound
    result['all_recorded_source_hashes_match_current_files'] = all(
        row['all_recorded_hashes_match_current'] for row in evidence.bindings)
    result['all_recorded_source_hashes_match_current_or_saved_snapshots'] = all(
        row['all_recorded_hashes_match_current_or_snapshot'] for row in evidence.bindings)
    result['identity_audit_limit'] = ('Matching hashes establish present file consistency with recorded identities, '
        'not authenticity of past execution or numerical correctness. Missing bindings are not inferred.')
    require_private_path_free(result)
    evidence.stable()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': OUTPUT.relative_to(BASE).as_posix(),
        'files_hashed': len(evidence.observed), 'reported_file_identities': len(evidence.files),
        'source_binding_groups': len(evidence.bindings),
        'all_recorded_source_hashes_match_current_files': result['all_recorded_source_hashes_match_current_files'],
        'all_recorded_source_hashes_match_current_or_saved_snapshots': result['all_recorded_source_hashes_match_current_or_saved_snapshots'],
        'ready_to_resume_recognition_training': False, 'verified_recognition_fix': False}))


if __name__ == '__main__':
    main()
