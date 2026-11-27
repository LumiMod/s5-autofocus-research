#!/usr/bin/env python3
"""Keep v1 train/validation and freeze a genuinely new, unscored test subset.

Only dataset images/metadata are accessed. No model, predictions or training run
is evaluated. Old test images are excluded because their results informed v2.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image

from recognition_data import (
    digest, duplicate_reason, image_fingerprints, load_pool, materialize,
    person_subclasses, rank, save_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def link_verified(source: Path, destination: Path, expected_sha: str):
    """Immutable local dataset copies share storage where the filesystem permits."""
    require(source.is_file() and not source.is_symlink(), 'Invalid source file: ' + str(source))
    require(digest(source) == expected_sha, 'Source changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    require(not destination.is_symlink() and digest(destination) == expected_sha,
            'Destination identity differs: ' + str(destination))


def confined(root: Path, name: str) -> Path:
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe dataset path')
    target = root / relative
    require(target.resolve().is_relative_to(root), 'Dataset path escapes its root')
    return target


def prior_conflict(frame: dict, prior: list[dict], distance: int) -> tuple[str, str] | None:
    for old in prior:
        if frame['source_image_id'] == old['source_image_id']:
            return 'old_source_image_id', old['id']
        if frame['group_id'] == old['group_id']:
            return 'old_author_group', old['id']
        reason = duplicate_reason(frame, old, distance)
        if reason:
            return 'old_' + reason, old['id']
    return None


def verify_novelty(new_test: list[dict], prior: list[dict], distance: int) -> dict:
    minimum_distance = 64
    for index, frame in enumerate(new_test):
        require(frame['split'] == frame['source_official_split'] == 'test', 'Expected official-test image')
        conflict = prior_conflict(frame, prior, distance)
        require(conflict is None, 'New test overlaps prior exposure: ' + str(conflict))
        for old in prior:
            minimum_distance = min(minimum_distance,
                                   (int(frame['dhash'], 16) ^ int(old['dhash'], 16)).bit_count())
        for other in new_test[:index]:
            require(frame['source_image_id'] != other['source_image_id'], 'Repeated new test ID')
            require(duplicate_reason(frame, other, distance) is None, 'Repeated new test content')
    return {'previous_exposure_images': len(prior), 'new_test_images': len(new_test),
            'old_ids_authors_file_and_pixel_hashes_disjoint': True,
            'old_vs_new_minimum_dhash_hamming_distance': minimum_distance,
            'dhash_reject_at_most': distance,
            'new_test_predictions_or_metrics_used': False,
            'scope': 'No exact/author/dHash overlap found; not exhaustive scene or pretraining-corpus independence.'}


def prepare(args):
    from pretraining_readiness import validate_recognition_dataset

    started = time.monotonic()
    previous = args.previous.resolve()
    output = args.output.resolve()
    require(previous != output and not output.is_relative_to(previous), 'Use a separate v2 directory')
    require(not (output / 'manifest.json').exists(), 'v2 manifest already exists; keep the frozen data unchanged')
    require(1 <= args.test <= 500 and 1 <= args.workers <= 8, 'Invalid bounded acquisition size/workers')
    require(args.max_per_author >= 1 and 0 <= args.dhash_distance <= 8, 'Invalid deduplication limits')
    previous_manifest = previous / 'manifest.json'
    previous_bytes = previous_manifest.read_bytes()
    previous_sha = hashlib.sha256(previous_bytes).hexdigest()
    previous_check, old = validate_recognition_dataset(previous_manifest)
    require(not previous_check['data_gaps'], 'Previous dataset is incomplete')
    prior = old['frames']
    kept = [copy.deepcopy(f) for f in prior if f['split'] in ('train', 'validation')]
    require(Counter(f['split'] for f in kept) == {'train': 320, 'validation': 80},
            'Expected the frozen 320 train / 80 validation sample')
    require(len(prior) == 500, 'Expected all 500 previously exposed images')
    provenance_path = confined(previous, old['provenance']['path'])
    previous_provenance_bytes = provenance_path.read_bytes()
    previous_provenance = json.loads(previous_provenance_bytes)
    require(digest(provenance_path) == old['provenance']['sha256'], 'Previous provenance changed')
    seed = previous_provenance['selection']['seed']
    output.mkdir(parents=True, exist_ok=True)
    # Recompute stored fingerprints before using them to exclude prior exposure.
    for frame in prior:
        with Image.open(confined(previous, frame['path'])) as image:
            pixels, dhash = image_fingerprints(image)
        require((pixels, dhash) == (frame['pixel_sha256'], frame['dhash']),
                'Stored prior image fingerprints differ: ' + frame['id'])
    for frame in kept:
        link_verified(confined(previous, frame['path']), output / frame['path'], frame['sha256'])
    for name, record in previous_provenance['sources'].items():
        link_verified(confined(previous, 'raw/' + name), output / 'raw' / name, record['sha256'])
    ancestry = output / 'ancestry'
    ancestry.mkdir(exist_ok=True)
    (ancestry / 'v1-manifest.json').write_bytes(previous_bytes)
    (ancestry / 'v1-provenance.json').write_bytes(previous_provenance_bytes)
    people = person_subclasses(json.loads((output / 'raw/hierarchy.json').read_text()))
    pool, pool_stats = load_pool(output, 'test', people, seed)
    pool.sort(key=lambda candidate: rank(seed, candidate['image_id']))
    old_ids = {f['source_image_id'] for f in prior}
    old_authors = {f['group_id'] for f in prior}
    exclusions = []
    candidates = []
    for candidate in pool:
        if candidate['image_id'] in old_ids or candidate['group_id'] in old_authors:
            exclusions.append({'image_id': candidate['image_id'], 'reason': 'previous_id_or_author'})
        else:
            candidates.append(candidate)
    new_test = []
    authors = Counter()
    position = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(new_test) < args.test and position < len(candidates):
            batch, pending_authors = [], Counter()
            while position < len(candidates) and len(batch) < min(args.workers, args.test - len(new_test)):
                candidate = candidates[position]
                position += 1
                group = candidate['group_id']
                if authors[group] + pending_authors[group] >= args.max_per_author:
                    exclusions.append({'image_id': candidate['image_id'], 'reason': 'new_test_author_cap'})
                    continue
                batch.append(candidate)
                pending_authors[group] += 1
            futures = [executor.submit(materialize, output, candidate, people) for candidate in batch]
            for candidate, future in zip(batch, futures):
                try:
                    frame = future.result()
                except Exception as exc:
                    exclusions.append({'image_id': candidate['image_id'], 'reason': 'materialization: ' + str(exc)})
                    continue
                conflict = prior_conflict(frame, prior, args.dhash_distance)
                if conflict is None:
                    for other in new_test:
                        reason = duplicate_reason(frame, other, args.dhash_distance)
                        if reason:
                            conflict = ('new_test_' + reason, other['id'])
                            break
                if conflict:
                    exclusions.append({'image_id': frame['source_image_id'],
                                       'reason': conflict[0], 'duplicate_of': conflict[1]})
                    continue
                new_test.append(frame)
                authors[frame['group_id']] += 1
            print(f'new_unscored_test_images_ready {len(new_test)} / {args.test}', flush=True)
    require(len(new_test) == args.test, 'Insufficient independent test candidates; preserve partial downloads')
    novelty = verify_novelty(new_test, prior, args.dhash_distance)
    frames = kept + new_test
    counts = dict(Counter(f['split'] for f in frames))
    provenance = copy.deepcopy(previous_provenance)
    provenance['revision'] = 'recognition-v2-new-unscored-test'
    provenance['previous_exposure'] = {
        'manifest_path': 'ancestry/v1-manifest.json', 'manifest_sha256': previous_sha,
        'provenance_path': 'ancestry/v1-provenance.json',
        'provenance_sha256': hashlib.sha256(previous_provenance_bytes).hexdigest(),
        'reason': 'v1 test outputs were inspected during diagnosis; those 100 images are no longer an independent v2 acceptance set.',
        'preserved': 'Exactly the original 320 train and 80 validation frame records, labels and image bytes.',
        'excluded_from_new_test': 'All previous 500 image IDs, authors, exact file/pixel hashes and dHash distance <= threshold.',
    }
    provenance['selection'].update(counts=counts, max_images_per_author=args.max_per_author,
        split_method='Frozen v1 train/validation; new official-test images in original author-test bucket, ordered by SHA256(seed:image_id), excluding all previous exposure before acceptance.',
        metadata_pool_stats={'test': pool_stats})
    provenance['duplicate_audit'] = {**novelty, 'excluded': exclusions,
        'source_group_count': len({f['group_id'] for f in frames}), 'cross_split_source_groups': 0}
    provenance['preparation_source_sha256'] = {
        name: digest(Path(__file__).resolve().parent / name)
        for name in ('prepare_recognition_v2_data.py', 'recognition_data.py')}
    provenance['limitations'].append('Train/validation have been inspected previously; only the new test is reserved for final v2 acceptance. This positive-only set still has no empty/background images.')
    save_json(output / 'provenance.json', provenance)
    manifest = {'schema_version': 1, 'task': 'recognition_detection', 'classes': old['classes'],
                'provenance': {'path': 'provenance.json', 'sha256': digest(output / 'provenance.json')},
                'frames': frames}
    pending = output / 'manifest.pending.json'
    save_json(pending, manifest)
    validation, _ = validate_recognition_dataset(pending)
    require(not validation['data_gaps'], 'Final v2 structural coverage gaps')
    require(previous_manifest.read_bytes() == previous_bytes, 'v1 manifest changed during acquisition')
    pending.replace(output / 'manifest.json')
    save_json(output / 'validation.json', validation)
    summary = {'manifest_sha256': digest(output / 'manifest.json'), 'counts': counts,
        'boxes': {split: dict(Counter(b['class'] for f in frames if f['split'] == split for b in f['boxes']))
                  for split in ('train', 'validation', 'test')},
        'source_groups': {split: len({f['group_id'] for f in frames if f['split'] == split})
                          for split in ('train', 'validation', 'test')},
        'previous_manifest_sha256': previous_sha, 'preserved_train_validation_records_identical': True,
        'novelty_audit': novelty, 'new_test_image_bytes': sum((output / f['path']).stat().st_size for f in new_test),
        'elapsed_seconds': time.monotonic() - started, 'exclusion_count': len(exclusions),
        'metadata_sources': provenance['sources'], 'training_or_prediction_run': False}
    save_json(output / 'data-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('private-data/recognition-v1'))
    parser.add_argument('--output', type=Path, default=Path('private-data/recognition-v2'))
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--max-per-author', type=int, default=2)
    parser.add_argument('--dhash-distance', type=int, default=4)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
