#!/usr/bin/env python3
"""Package the frozen AF work without environments, original photos or full firmware."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact_policy(relative):
    parts = relative.parts
    if '__pycache__' in parts or relative.suffix in ('.pyc', '.pyo') or relative.name == '.DS_Store':
        return False, 'generated Python/macOS cache'
    if parts[:2] == ('external-baselines', 'cache'):
        return False, 'redundant third-party download cache'
    if parts[:2] == ('external-baselines', 'vendor'):
        if (relative.name == 'LICENSE' or relative.as_posix().endswith('/deploy/python/preprocess.py')
                or relative.as_posix().endswith('/deploy/python/keypoint_preprocess.py')):
            return True, 'minimal third-party adapter dependency and license'
        return False, 'third-party source checkout; reconstruct pinned commit'
    if len(parts) == 2 and parts[0] in ('43_hm_d_nw', '44_hm_d_nw_sng') and (
            relative.suffix == '.bin' or relative.name == 'numeric_tensors.npz'):
        return False, 'original full component or extracted original parameter store; regenerate from external firmware'
    if parts[0] == 'photo-probe' and relative.name == 'prepared_input.npz':
        return False, 'photo input pixels'
    if relative.suffix.lower() == '.pdf':
        return False, 'external reference PDF; source URL and SHA retained'
    return True, 'research artifact'


def run(args):
    repo, source = args.repo.resolve(), args.original_af.resolve()
    output = repo/'.handoff-local'/args.archive_name
    assert not output.exists(), 'Keep previous archive immutable'
    assert (repo/'af-research/docs/AF_MODEL_HANDOFF_2026-11-27.md').is_file(), 'Missing handoff entry point'
    included, excluded = {}, []
    for path in sorted(repo.rglob('*')):
        relative = path.relative_to(repo)
        if any(p in ('.git', '.handoff-local', '__pycache__') for p in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError('Unexpected source symlink: '+str(relative))
        if path.is_file() and relative.as_posix() not in ('handoff/MANIFEST.json', 'handoff/PACKAGE_SCOPE.json'):
            included[relative.as_posix()] = (path, 'repository source/document')
    for path in sorted((source/'artifacts').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source/'artifacts')
        keep, reason = artifact_policy(relative)
        entry = 'af-research/artifacts/'+relative.as_posix()
        if keep:
            assert not path.is_symlink(), 'Unexpected included artifact symlink'
            included[entry] = (path, reason)
        else:
            excluded.append(dict(path=entry, reason=reason, bytes=path.lstat().st_size))
    metadata_suffixes = {'.json', '.jsonl', '.csv', '.log', '.txt', '.md'}
    for path in sorted((source/'private-data').rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source/'private-data')
        if any(p in ('images', '__pycache__') for p in relative.parts):
            continue
        if path.suffix.lower() in metadata_suffixes:
            included['af-research/private-data/'+relative.as_posix()] = (path, 'data annotation/provenance metadata; no original photo')
    rows = [dict(path=name, bytes=path.stat().st_size, sha256=digest(path), category=category)
            for name, (path, category) in sorted(included.items())]
    counts = Counter(row['category'] for row in rows)
    scope = dict(date='2026-11-27', repository='LumiMod/s5-autofocus-research', visibility='private',
        archive=args.archive_name, included_file_count=len(rows), included_bytes=sum(r['bytes'] for r in rows),
        categories=dict(counts), excluded_artifact_files=excluded,
        other_not_packaged=['Python environments', 'original dataset photographs', 'full original firmware outside af-research',
                            'QEMU/compiler toolchain binaries', 'Git credentials and repository internals'],
        original_firmware_fragments_included='Small ARM converter instruction slices are retained as replay evidence; full components are excluded.',
        experimental_checkpoints_included=True, trained_model_accepted=False, ready_to_train=False,
        historical_paths='Historical reports and source snapshots keep absolute development paths; use documented explicit-argument entry points on a new machine.',
        dataset_test_scope='Label/provenance files are packaged without model inference; held-out test pictures are excluded.')
    manifest = dict(schema_version=1, hash='sha256', files=rows,
                    self_reference='MANIFEST.json and PACKAGE_SCOPE.json are included in ZIP but omitted from this self-hash list; ZIP has external SHA256SUMS.')
    handoff = repo/'handoff'
    handoff.mkdir(exist_ok=True)
    (handoff/'MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    (handoff/'PACKAGE_SCOPE.json').write_text(json.dumps(scope, indent=2, ensure_ascii=False)+'\n')
    included['handoff/MANIFEST.json'] = (handoff/'MANIFEST.json', 'package manifest')
    included['handoff/PACKAGE_SCOPE.json'] = (handoff/'PACKAGE_SCOPE.json', 'package scope')
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name, (path, _) in sorted(included.items()):
            archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        assert set(archive.namelist()) == set(included), 'Missing or extra ZIP entry'
        for row in rows:
            h = hashlib.sha256()
            with archive.open(row['path']) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
            assert h.hexdigest() == row['sha256'], 'Archived content changed: '+row['path']
    sha = digest(output)
    (output.parent/'SHA256SUMS.txt').write_text(sha+'  '+output.name+'\n')
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size,
                         entries=len(included), sha256=sha, crc_and_all_file_hashes_passed=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--original-af', type=Path, required=True)
    p.add_argument('--archive-name', default='S5_AF_Research_Handoff_2026-11-27.zip')
    run(p.parse_args())
