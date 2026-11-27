#!/usr/bin/env python3
"""Inspect ten 32-bit slots around known S5 quantization-scale candidates.

Raw bits and IEEE754 interpretations are evidence; original C types, consumers
and runtime use are not established. No network or photograph is executed.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct

from decode_parameters import recover, tail_offset
from extract import MODELS, decode, digest, require
from research_paths import add_firmware_arguments, resolve_source


BASE = Path(__file__).resolve().parent


def slots(data, offset):
    blob = data[offset:offset + 40]
    require(len(blob) == 40, "Slot extent overflow")
    words = struct.unpack("<10I", blob)
    floats = struct.unpack("<10f", blob)
    extra = set(words[1:])
    if extra == {0}:
        pattern = "nine_zero_words"
    elif extra == {words[0]}:
        pattern = "nine_bitwise_copies_of_primary"
    elif extra == {0x3f800000}:
        pattern = "nine_IEEE754_float_ones"
    elif extra == {0x40000000}:
        pattern = "nine_IEEE754_float_twos"
    elif extra == {0xffc00000}:
        pattern = "nine_IEEE754_negative_quiet_NaNs"
    else:
        pattern = "other"
    return {"file_offset": offset, "bytes": 40, "sha256": digest(blob),
            "raw_hex": blob.hex(), "u32_hex": [f"0x{x:08x}" for x in words],
            "ieee754_f32_interpretation": [float(v) if math.isfinite(v) else "NaN" for v in floats],
            "extra_nine_pattern": pattern}


def inspect(data):
    graph = decode(data)
    params, _ = recover(data)
    outputs = []
    descriptors = []
    duplicates = []
    for node in graph["layers"]:
        q = tail_offset(node)
        offset = q + {10: 19, 8: 24, 24: 11}.get(node["operator_id"], 7)
        row = {"layer": node["index"], "name": node["name"], "operator_id": node["operator_id"],
               "inputs": node["inputs"], "block": slots(data, offset)}
        outputs.append(row)
        if node["operator_id"] in (1, 5):
            duplicate_offset = q + (255 if node["operator_id"] == 1 else 90)
            same = data[offset:offset + 40] == data[duplicate_offset:duplicate_offset + 40]
            require(same, "Convolution/pool duplicate numeric block changed")
            duplicates.append({"layer": node["index"], "name": node["name"],
                               "first_offset": offset, "second_offset": duplicate_offset,
                               "all40_bytes_equal": same,
                               "second_block": slots(data, duplicate_offset)})
    for record in params["convolutions"]:
        for kind in ("weight", "bias"):
            meta = record[kind]
            descriptors.append({"layer": record["index"], "name": record["name"], "kind": kind,
                                "descriptor_offset": meta["descriptor_offset"],
                                "descriptor_stored_shape": meta["stored_shape"],
                                "block": slots(data, meta["descriptor_offset"] + 28)})
    groups = Counter((row["operator_id"], row["block"]["extra_nine_pattern"]) for row in outputs)
    patterns = Counter(row["block"]["extra_nine_pattern"] for row in descriptors)
    by_layer = {row["layer"]: row for row in outputs}
    slice_copies = []
    for row in outputs:
        if row["operator_id"] != 10:
            continue
        predecessor = by_layer[row["inputs"][0]["layer"]]
        slice_copies.append({"slice": row["name"], "producer": predecessor["name"],
                             "all40_bytes_identical": row["block"]["raw_hex"] == predecessor["block"]["raw_hex"],
                             "primary_f32_candidate": row["block"]["ieee754_f32_interpretation"][0]})
    return {"source_sha256": digest(data), "outputs": outputs, "parameter_descriptors": descriptors,
            "output_pattern_counts": [{"operator_id": key[0], "pattern": key[1], "count": count}
                                      for key, count in sorted(groups.items())],
            "descriptor_pattern_counts": dict(patterns),
            "convolution_pool_duplicate_blocks": duplicates,
            "slice_producer_numeric_block_copies": slice_copies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument("--output", type=Path, default=BASE / "artifacts/recognition-diagnosis-v3/static-constraints/quant-metadata.json")
    args = parser.parse_args()
    source = resolve_source(args)
    models = {}
    for stem, expected in MODELS.items():
        data = (source / (stem + ".bin")).read_bytes()
        require(digest(data) == expected, "Wrong original component")
        models[stem] = inspect(data)
    report = {"schema_version": 1, "status": "numeric metadata patterns constrained; original field types and execution unresolved",
              "scope": {"components": list(MODELS), "images_read": 0, "inference_runs": 0, "optimizer_steps": 0},
              "source_hashes": {name: digest((BASE / name).read_bytes()) for name in
                                [Path(__file__).name, "extract.py", "decode_parameters.py"]},
              "observations": [
                  "Each model has11 uconv output blocks and11 corresponding slice blocks with all10 words bitwise equal to a finite fractional primary-scale candidate.",
                  "Classification logits and their reshape records have nine0x3f800000 words; softmax, its reshape and bbox records have nine0x40000000 words.",
                  "Ordinary weight/bias descriptor extra slots are zero, except both upsample descriptors contain nine0xffc00000 words interpreted as IEEE754 quiet NaNs; their first slots are finite.",
                  "All convolution/pool40-byte output numeric blocks repeat exactly later in their records; each of11 slice blocks copies its uconv producer's complete40-byte block.",
                  "All selected records retain exact original offsets, raw40-byte blocks and both word/IEEE754 interpretations."],
              "inferences": [
                  "The nine slots are not consistently represented as ordinary integer/Boolean0-or1 flags: fractional repeated scales, float2.0 patterns and NaN patterns contradict that uniform interpretation.",
                  "They cannot all be unconditional multiplicative network coefficients: ordinary layers would be zeroed and upsample would propagateNaNs.",
                  "A fixed-width numeric metadata area with context-dependent valid/default/unused entries is compatible with these bytes, but is not a recovered ABI."],
              "unknowns": [
                  "No ARM/XM4 consumer has been located for these exact fields, so C type, valid-entry count and per-port/per-channel use are unknown.",
                  "The first-slot scale relationship is supported independently by input_scale*weight_scale=bias_scale; that does not assign meanings to remaining slots.",
                  "No evidence here authorizes multiplying weights/activations by these nine fields or treating them as active normalization/activation parameters."],
              "models": models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": digest(args.output.read_bytes()),
                      "script_sha256": report["source_hashes"][Path(__file__).name]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
