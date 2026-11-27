#!/usr/bin/env python3
"""Acquire a small, source-separated real Open Images person/face pilot.

Only public dataset bytes are downloaded, never user photographs or firmware.
The output directory is local/ignored. This is a deliberately positive-only
pilot, not the official Open Images benchmark or a test of physical autofocus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request
import urllib.parse

from PIL import Image

BASE = "https://storage.googleapis.com/openimages"
SOURCES = {
    "classes.csv": f"{BASE}/v7/oidv7-class-descriptions-boxable.csv",
    "hierarchy.json": f"{BASE}/2018_04/bbox_labels_600_hierarchy.json",
    "validation-bbox.csv": f"{BASE}/v5/validation-annotations-bbox.csv",
    "test-bbox.csv": f"{BASE}/v5/test-annotations-bbox.csv",
    "validation-images.csv": f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
    "test-images.csv": f"{BASE}/2018_04/test/test-images-with-rotation.csv",
}
PERSON = "/m/01g317"
FACE = "/m/0dzct"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(url: str, path: Path, max_bytes: int = 150_000_000) -> None:
    """Bound each public download and atomically replace incomplete files."""
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "S5-recognition-research/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, part.open("wb") as output:
                total = 0
                while block := response.read(1 << 18):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError(f"Download exceeds {max_bytes} bytes: {url}")
                    output.write(block)
            part.replace(path)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def source_group(metadata: dict) -> str | None:
    # One photographer can upload a burst/sequence under distinct image IDs.
    # Split by the public author profile, more conservatively than by image ID.
    profile = metadata.get("AuthorProfileURL", "").strip().lower().rstrip("/")
    if not profile:
        return None
    profile = profile.replace("http://", "https://").replace("www.flickr.com", "flickr.com")
    return "flickr-author-" + hashlib.sha256(profile.encode()).hexdigest()[:24]


def rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_split(group: str, seed: int) -> str:
    bucket = int(rank(seed, group)[:8], 16) % 100
    return "train" if bucket < 64 else "validation" if bucket < 80 else "test"


def person_subclasses(hierarchy: dict) -> set[str]:
    def descendants(node: dict) -> set[str]:
        return {node["LabelName"]}.union(*(descendants(n) for n in node.get("Subcategory", [])))
    def find(node: dict) -> set[str]:
        if node["LabelName"] == PERSON:
            return descendants(node)
        for key in ("Subcategory", "Part"):
            for child in node.get(key, []):
                result = find(child)
                if result:
                    return result
        return set()
    result = find(hierarchy)
    if PERSON not in result:
        raise ValueError("Person hierarchy not found")
    return result


def ambiguous_person_subclasses(rows: list[dict], person_labels: set[str]) -> bool:
    """Reject conflicting boxes, rather than invent a merged target boundary.

    Open Images can box one person as both Girl/Woman or Boy/Man with slightly
    different boundaries. After collapsing classes these become duplicate GT.
    Cross-source-class IoU >= .5 OR intersection/min(area) >= .95 is treated as
    ambiguous. This may also exclude some genuine nested/occluding people.
    """
    people = [r for r in rows if r["LabelName"] in person_labels]
    for index, first in enumerate(people):
        a = [float(first[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        for second in people[index + 1:]:
            if first["LabelName"] == second["LabelName"]:
                continue
            b = [float(second[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
            if min(area_a, area_b) <= 0:
                return True
            if overlap / (area_a + area_b - overlap) >= .5 or overlap / min(area_a, area_b) >= .95:
                return True
    return False


def load_pool(root: Path, official_split: str, person_labels: set[str], seed: int) -> tuple[list[dict], dict]:
    annotations: dict[str, list] = defaultdict(list)
    rejected: set[str] = set()
    target_labels = person_labels | {FACE}
    counters: Counter = Counter()
    with (root / "raw" / f"{official_split}-bbox.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["LabelName"] not in target_labels:
                continue
            image_id = row["ImageID"]
            counters["target_boxes_read"] += 1
            # Exclude entire image, rather than count discarded group/depiction
            # boxes as unlabeled negative background during training/evaluation.
            if int(row["IsGroupOf"]) or int(row["IsDepiction"]):
                rejected.add(image_id)
            annotations[image_id].append(row)
    pool = []
    with (root / "raw" / f"{official_split}-images.csv").open(newline="") as stream:
        for metadata in csv.DictReader(stream):
            image_id = metadata["ImageID"]
            if image_id not in annotations or image_id in rejected:
                continue
            labels = {row["LabelName"] for row in annotations[image_id]}
            if FACE not in labels or not (labels & person_labels):
                continue
            if ambiguous_person_subclasses(annotations[image_id], person_labels):
                counters["ambiguous_overlapping_person_subclasses_excluded"] += 1
                continue
            group = source_group(metadata)
            if group is None:
                counters["missing_author_excluded"] += 1
                continue
            license_url = metadata.get("License", "").replace("http://", "https://").rstrip("/")
            if license_url != "https://creativecommons.org/licenses/by/2.0":
                counters["other_or_missing_license_excluded"] += 1
                continue
            # CVDF pixels are stored unrotated: avoid ambiguous nonzero rotation
            # rather than silently applying EXIF or metadata to bounding boxes.
            rotation = metadata.get("Rotation", "").strip()
            if rotation not in ("", "0", "0.0"):
                counters["nonzero_rotation_excluded"] += 1
                continue
            split = group_split(group, seed)
            if (official_split == "test") != (split == "test"):
                continue
            pool.append({"image_id": image_id, "official_split": official_split,
                         "split": split, "group_id": group, "metadata": metadata,
                         "annotations": annotations[image_id]})
    pool.sort(key=lambda x: rank(seed, x["image_id"]))
    counters["images_with_target_group_or_depiction_excluded"] = len(rejected)
    counters["eligible_images"] = len(pool)
    return pool, dict(counters)


def image_fingerprints(image: Image.Image) -> tuple[str, str]:
    rgb = image.convert("RGB")
    pixels = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    small = list(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(small[y * 9 + x + 1] > small[y * 9 + x])
    return pixels, f"{bits:016x}"


def materialize(root: Path, candidate: dict, person_labels: set[str]) -> dict:
    image_id = candidate["image_id"]
    split = candidate["official_split"]
    relative = f"images/{split}/{image_id}.jpg"
    path = root / relative
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    download(url, path, max_bytes=8_000_000)
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        orientation = image.getexif().get(274, 1)
        if orientation not in (None, 1):
            raise ValueError(f"Nontrivial EXIF orientation {orientation}; excluded")
        pixels, dhash = image_fingerprints(image)
    boxes = []
    seen = set()
    for row in candidate["annotations"]:
        label = "person" if row["LabelName"] in person_labels else "person_face"
        normalized = tuple(float(row[key]) for key in ("XMin", "YMin", "XMax", "YMax"))
        if not (0 <= normalized[0] < normalized[2] <= 1 and 0 <= normalized[1] < normalized[3] <= 1):
            raise ValueError("Invalid normalized source box")
        identity = (label, normalized)
        if identity in seen:
            continue
        seen.add(identity)
        boxes.append({"class": label,
                      "bbox_xyxy": [normalized[0] * width, normalized[1] * height,
                                    normalized[2] * width, normalized[3] * height],
                      "source_label": row["LabelName"],
                      "attributes": {key: int(row[key]) for key in
                                     ("IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside")}})
    metadata = candidate["metadata"]
    return {"id": f"oi-{split}-{image_id}", "path": relative,
            "sha256": digest(path), "pixel_sha256": pixels, "dhash": dhash,
            "width": width, "height": height, "split": candidate["split"],
            "source_kind": "public_dataset", "source_dataset": "Open Images V7",
            "source_image_id": image_id, "group_id": candidate["group_id"],
            "source_official_split": split, "source_url": metadata["OriginalLandingURL"],
            "download_url": url, "license_url": metadata["License"],
            "source_author": metadata["Author"], "source_author_profile": metadata["AuthorProfileURL"],
            "source_title": metadata["Title"], "boxes": boxes}


def duplicate_reason(frame: dict, existing: dict, max_hamming: int) -> str | None:
    if frame["sha256"] == existing["sha256"]:
        return "identical_file_sha256"
    if frame["pixel_sha256"] == existing["pixel_sha256"]:
        return "identical_decoded_pixels"
    if (int(frame["dhash"], 16) ^ int(existing["dhash"], 16)).bit_count() <= max_hamming:
        return "dhash_hamming_le_" + str(max_hamming)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("private-data/recognition-v1"))
    parser.add_argument("--train", type=int, default=320)
    parser.add_argument("--validation", type=int, default=80)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-per-author", type=int, default=2)
    parser.add_argument("--dhash-distance", type=int, default=4)
    parser.add_argument("--from-manifest", type=Path,
                        help="Replay a frozen manifest and verify each exact downloaded SHA-256")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.from_manifest:
        frozen_path = args.from_manifest.resolve()
        frozen = json.loads(frozen_path.read_text())
        provenance_record = frozen["provenance"]
        provenance_relative = Path(provenance_record["path"])
        if provenance_relative.is_absolute() or ".." in provenance_relative.parts:
            raise ValueError("Unsafe frozen provenance path")
        provenance_path = frozen_path.parent / provenance_relative
        if digest(provenance_path) != provenance_record["sha256"]:
            raise ValueError("Frozen provenance SHA-256 mismatch")
        for frame in frozen["frames"]:
            relative = Path(frame["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe frozen image path")
            parsed = urllib.parse.urlparse(frame["download_url"])
            if parsed.scheme != "https" or parsed.netloc != "open-images-dataset.s3.amazonaws.com":
                raise ValueError("Frozen download URL is not the expected public image bucket")
            download(frame["download_url"], root / relative, max_bytes=8_000_000)
            if digest(root / relative) != frame["sha256"]:
                raise ValueError(f"Frozen image SHA-256 mismatch: {frame['id']}")
        (root / "provenance.json").write_bytes(provenance_path.read_bytes())
        (root / "manifest.json").write_bytes(frozen_path.read_bytes())
        print(f"Verified {len(frozen['frames'])} frozen images; manifest SHA256 {digest(root / 'manifest.json')}")
        return
    goals = {name: getattr(args, name) for name in ("train", "validation", "test")}
    if any(n < 1 for n in goals.values()) or sum(goals.values()) > 2000:
        raise ValueError("Each split needs >=1 image; pilot limited to 2000 total")
    if not 1 <= args.workers <= 12 or args.max_per_author < 1 or not 0 <= args.dhash_distance <= 8:
        raise ValueError("Invalid resource or deduplication limits")
    started = time.monotonic()
    def get_source(item: tuple[str, str]) -> None:
        name, url = item
        download(url, root / "raw" / name)
        print(f"metadata_ready {name}", flush=True)
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as executor:
        list(executor.map(get_source, SOURCES.items()))
    people = person_subclasses(json.loads((root / "raw/hierarchy.json").read_text()))
    pools = []
    pool_stats = {}
    for official in ("validation", "test"):
        pool, stats = load_pool(root, official, people, args.seed)
        pools.extend(pool)
        pool_stats[official] = stats
    # Freeze candidate selection order before any model evaluation takes place.
    pools.sort(key=lambda x: (x["split"], rank(args.seed, x["image_id"])))
    frames: list[dict] = []
    counts: Counter = Counter()
    authors: Counter = Counter()
    excluded = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for target in ("train", "validation", "test"):
            candidates = [x for x in pools if x["split"] == target]
            position = 0
            while counts[target] < goals[target] and position < len(candidates):
                batch = []
                pending_authors = Counter()
                while position < len(candidates) and len(batch) < min(args.workers, goals[target] - counts[target]):
                    candidate = candidates[position]
                    position += 1
                    group = candidate["group_id"]
                    if authors[group] + pending_authors[group] >= args.max_per_author:
                        continue
                    batch.append(candidate)
                    pending_authors[group] += 1
                futures = [executor.submit(materialize, root, item, people) for item in batch]
                for candidate, future in zip(batch, futures):
                    try:
                        frame = future.result()
                    except Exception as exc:
                        excluded.append({"image_id": candidate["image_id"], "reason": str(exc)})
                        continue
                    duplicate = next(((other, reason) for other in frames
                                      if (reason := duplicate_reason(frame, other, args.dhash_distance))), None)
                    if duplicate:
                        other, reason = duplicate
                        excluded.append({"image_id": candidate["image_id"], "reason": reason,
                                         "duplicate_of": other["id"], "candidate_split": target,
                                         "existing_split": other["split"]})
                        continue
                    frames.append(frame)
                    counts[target] += 1
                    authors[frame["group_id"]] += 1
                print(f"images_ready {dict(counts)}", flush=True)
            if counts[target] != goals[target]:
                raise RuntimeError(f"Insufficient eligible {target} images: {counts[target]} / {goals[target]}")
    frame_groups: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        frame_groups[frame["group_id"]].add(frame["split"])
    assert all(len(splits) == 1 for splits in frame_groups.values())
    source_records = {name: {"url": url, "sha256": digest(root / "raw" / name),
                             "bytes": (root / "raw" / name).stat().st_size}
                      for name, url in SOURCES.items()}
    provenance = {
        "dataset": "Open Images V7, V5 bounding boxes linked by the V7 official download page",
        "official_documentation": "https://storage.googleapis.com/openimages/web/download_v7.html",
        "official_licenses_and_annotation_scope": "https://storage.googleapis.com/openimages/web/factsfigures_v7.html",
        "official_image_downloader": "https://github.com/openimages/dataset/blob/master/downloader.py",
        "citation": "Kuznetsova et al., The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale, IJCV 2020.",
        "annotation_license": "https://creativecommons.org/licenses/by/4.0/",
        "image_license": "https://creativecommons.org/licenses/by/2.0/",
        "license_verification": "Per-image license metadata retained from official CSV; original Flickr license status not independently re-audited. Google disclaims individual-image license warranties. No image redistribution in the source repository.",
        "sources": source_records,
        "selection": {"seed": args.seed, "counts": dict(counts), "max_images_per_author": args.max_per_author,
                      "person_label_mids": sorted(people), "face_label_mid": FACE,
                      "filter": "Both person (including official subclasses) and human face boxes required; exclude entire image if any target group-of/depiction box, unknown author, non-CC-BY-2.0 metadata, nonzero rotation or nontrivial EXIF orientation. Exclude ambiguous cross-subclass person boxes with IoU >= 0.5 OR intersection/min(area) >= 0.95 to avoid counting one person multiple times after class collapse; this may remove genuine nested/occluding people too. No minimum box size or appearance filter. Retain occluded/truncated/inside attributes. Deduplicate exactly equal mapped boxes.",
                      "overlapping_person_subclass_filter": {"iou_threshold": .5, "intersection_over_smaller_area_threshold": .95, "action": "exclude_whole_image", "reason": "QA found repeated Girl/Woman and Boy/Man boxes for the same visible person after mapping subclasses to person; a subsequent audit also found Girl/Person duplicates at IoU 0.64. No model score or prediction used in this correction."},
                      "split_method": "SHA256(seed:normalized-Flickr-author-group) bucket 0..63 train, 64..79 validation, 80..99 test. Train/validation use only official validation images; test uses only official test. Select SHA256(seed:image_id) order with max-per-author cap. This local train subset is not the official Open Images training split.",
                      "metadata_pool_stats": pool_stats},
        "duplicate_audit": {"file_sha256": True, "decoded_rgb_sha256": True,
                            "dhash": "64-bit horizontal dHash, grayscale 9x8 BILINEAR, right>left",
                            "dhash_hamming_threshold": args.dhash_distance,
                            "policy": "Reject subsequent near/exact duplicates anywhere in the selected set; author groups cannot cross splits.",
                            "source_group_count": len(frame_groups), "cross_split_source_groups": 0,
                            "excluded": excluded},
        "limitations": ["Positive-only small pilot; no image-level negative examples, so empty-scene false-positive performance is unmeasured.",
                        "All target instances are provided for available positive labels according to Open Images; missing labels and annotation errors remain possible. This is a two-class local AP50 evaluation, not official Open Images mAP.",
                        "Author grouping and perceptual hashes reduce adjacent-sequence/duplicate leakage; no capture-time metadata or exhaustive visual duplicate verification exists, so unknown cross-author reposts or transformed near-duplicates may remain.",
                        "Held-out independence is relative to this fine-tuning run. The proprietary factory model's original training corpus is unknown; overlap with that corpus cannot be excluded.",
                        "No S5 capture, measured light condition, focus-control trace or identity-recognition label. No low-light or camera deployment conclusion is supported."]}
    save_json(root / "provenance.json", provenance)
    manifest = {"schema_version": 1, "task": "recognition_detection",
                "classes": ["person", "person_face"],
                "provenance": {"path": "provenance.json", "sha256": digest(root / "provenance.json")},
                "frames": frames}
    save_json(root / "manifest.json", manifest)
    summary = {"manifest_sha256": digest(root / "manifest.json"), "counts": dict(counts),
               "boxes": {split: dict(Counter(box["class"] for frame in frames if frame["split"] == split
                                             for box in frame["boxes"])) for split in goals},
               "source_groups": {split: len({f["group_id"] for f in frames if f["split"] == split}) for split in goals},
               "downloaded_image_bytes": sum((root / f["path"]).stat().st_size for f in frames),
               "elapsed_seconds": time.monotonic() - started,
               "exclusion_count": len(excluded), "metadata_sources": source_records}
    save_json(root / "data-summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
