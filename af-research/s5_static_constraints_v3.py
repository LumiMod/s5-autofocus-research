#!/usr/bin/env python3
"""Bounded original-byte constraints for S5 bias, graph ports and GEMM K tiles.

Static only: no image loading, inference, loss, optimizer or AP selection.
The proposed K4 layout is a hypothesis supported by cross-layer sparsity;
its mathematical invertibility does not establish original DSP execution.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct

import numpy as np

from decode_parameters import recover, tail_offset
from extract import MODELS, decode, digest, require
from ops_investigation import analyze
from packing_investigation import unpack
from research_paths import add_firmware_arguments, resolve_source


BASE = Path(__file__).resolve().parent


def gemm_k4_weight(raw, outputs, inputs, kernel, order="IHW"):
    """Physical [O/4, K/4, O4, K4]; K is flattened I*H*W, not I4*H*W."""
    kh, kw = kernel
    total_k = inputs * kh * kw
    require(outputs % 4 == total_k % 4 == 0, "Nonintegral GEMM4 tile")
    flat = np.asarray(raw).reshape(outputs // 4, total_k // 4, 4, 4)
    flat = flat.transpose(0, 2, 1, 3).reshape(outputs, total_k)
    if order == "IHW":
        return flat.reshape(outputs, inputs, kh, kw).copy()
    require(order == "HWI", "Unknown K flatten order")
    return flat.reshape(outputs, kh, kw, inputs).transpose(0, 3, 1, 2).copy()


def gemm_k4_pack(weight):
    outputs = weight.shape[0]
    flat = np.asarray(weight).reshape(outputs, -1)
    return flat.reshape(outputs // 4, 4, flat.shape[1] // 4, 4).transpose(0, 2, 1, 3).copy()


def bytes_evidence(data, offset, count):
    blob = data[offset:offset + count]
    require(len(blob) == count, "Evidence extent overflow")
    return {"file_offset": offset, "bytes": count, "sha256": digest(blob), "raw_hex": blob.hex()}


def tensor_summary(weight):
    a = weight.astype(np.float64)
    return {
        "exact_zero_input_channels": np.flatnonzero(np.all(weight == 0, axis=(0, 2, 3))).tolist(),
        "exact_zero_output_channels": np.flatnonzero(np.all(weight == 0, axis=(1, 2, 3))).tolist(),
        "all_zero_channel_kernel_pairs": int(np.count_nonzero(np.all(weight == 0, axis=(2, 3)))),
        "input_rms": np.sqrt(np.mean(a * a, axis=(0, 2, 3))).tolist(),
        "output_rms": np.sqrt(np.mean(a * a, axis=(1, 2, 3))).tolist(),
    }


def producer_channels(index, port, analysis):
    """Resolve only pure channel slices to their immediate convolution producer."""
    node = analysis[index]
    if node["operator_id"] == 1:
        return index, list(range(node["output_channels"]))
    if node["operator_id"] == 10:
        edge = node["inputs"][0]
        origin, channels = producer_channels(edge["layer"], edge["port"], analysis)
        cut = node["cuts"][0]
        require(port in (0, 1), "Unexpected slice port")
        return origin, channels[:cut] if port == 0 else channels[cut:]
    return None, []


def inspect_model(data):
    graph = decode(data)
    params, tensors = recover(data)
    analysis = analyze(data)["layers"]
    records = {r["index"]: r for r in params["convolutions"]}
    weights, rows, bias_rows, heads = {}, [], [], []
    for r in params["convolutions"]:
        key = r["tensor_key"]
        raw = tensors[key + "_weight_i16_packed"]
        bias = tensors[key + "_bias_i32_pairs"]
        logical = r["output_channels_candidate"]
        bias_rows.append({
            "layer": r["index"], "name": r["name"], "offset": r["bias"]["data_offset"],
            "sha256": r["bias"]["sha256"], "logical_channels": logical,
            "stored_i32_pair_rows": int(bias.shape[0]),
            "all_adjacent_pairs_identical": bool(np.array_equal(bias[:, 0], bias[:, 1])),
            "padding_rows_all_zero": bool(np.all(bias[logical:] == 0)),
            "scale_identity_relative_error": r["scale_identity_relative_error"],
        })
        if r["name"].startswith("rpn_cls_score"):
            candidates = []
            for perm in itertools.permutations(range(4)):
                b = bias[:, 0].reshape(-1, 4)[:, perm].reshape(-1)
                residual = np.max(np.abs(b[:25].astype(np.int64) + b[25:50].astype(np.int64)))
                ratio = float(residual / max(1, np.max(np.abs(b[:50]))))
                if np.all(b[50:] == 0) and ratio < 1e-5:
                    candidates.append({"permutation": list(perm), "maximum_pair_relative_residual": ratio})
            heads.append({"layer": r["index"], "name": r["name"],
                          "bias_values_i32_one_per_pair": bias[:, 0].tolist(),
                          "passing_O4_lane_permutations": candidates})
        if r["name"] == "upsample":
            continue
        physical = r["weight"]["stored_shape"][0]
        inputs = analysis[r["index"]]["input_shapes_candidate"][0][1]
        w = gemm_k4_weight(raw, physical, inputs, r["kernel_hw_candidate"])
        require(np.array_equal(gemm_k4_pack(w).reshape(raw.shape), raw), "K4 roundtrip mismatch")
        require(np.all(w[logical:] == 0), "K4 output padding mismatch")
        weights[r["index"]] = w[:logical]
        row = {"layer": r["index"], "name": r["name"], "input_channels": inputs,
               "output_channels": logical, "physical_output_channels": physical,
               "kernel": r["kernel_hw_candidate"], "weight_data_offset": r["weight"]["data_offset"],
               "weight_bytes": r["weight"]["bytes"], "weight_sha256": r["weight"]["sha256"],
               "K": int(np.prod(w.shape[1:])), "roundtrip_identical": True,
               "output_padding_all_zero": True, "K4_IHW": tensor_summary(w[:logical]),
               "bias_f32": tensors[key + "_bias_f32"].astype(float).tolist()}
        row["K4_HWI_counterhypothesis"] = tensor_summary(
            gemm_k4_weight(raw, physical, inputs, r["kernel_hw_candidate"], "HWI")[:logical])
        if r["kernel_hw_candidate"] == [1, 1]:
            row["identical_to_existing_1x1_unpack"] = bool(np.array_equal(w[:logical], unpack(raw, r)))
        elif r["name"] == "conv1":
            how = raw.reshape(physical // 4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(physical, 1, 4, 4)
            row["identical_to_previous_H_O_W_conv1_candidate"] = bool(np.array_equal(w, how))
        else:
            row["legacy_O4_I4_HW"] = tensor_summary(unpack(raw, r, "o4_i4_hw"))
        rows.append(row)

    edge_constraints = []
    selected_names = {"conv3", "inc3d/conv5_2", "inc3e/conv3_2"}
    for r in params["convolutions"]:
        if r["name"] not in selected_names:
            continue
        edge = analysis[r["index"]]["inputs"][0]
        origin, mapping = producer_channels(edge["layer"], edge["port"], analysis)
        require(origin in records and len(mapping) == weights[r["index"]].shape[1], "Unexpected selected edge")
        source_r = records[origin]
        source_w = weights[origin].astype(np.float64) * source_r["weight_scale"]
        source_b = tensors[source_r["tensor_key"] + "_bias_f32"].astype(np.float64)
        dst_w = weights[r["index"]].astype(np.float64) * r["weight_scale"]
        dst_rms = np.sqrt(np.mean(dst_w ** 2, axis=(0, 2, 3)))
        source_rms = np.sqrt(np.mean(source_w ** 2, axis=(1, 2, 3)))
        candidates = set(np.flatnonzero(dst_rms == 0).tolist())
        candidates.update(sorted(range(len(mapping)), key=lambda j: source_rms[mapping[j]])[:2])
        selected = []
        for channel in sorted(candidates):
            sc = mapping[channel]
            rank = int(np.count_nonzero(dst_rms < dst_rms[channel])) + 1
            cap = 32767.0 * source_r["input_scale"]
            upper = float(source_b[sc] + cap * np.maximum(source_w[sc], 0).sum())
            selected.append({"consumer_input_channel": channel, "producer_output_channel": sc,
                             "producer_weight_rms": float(source_rms[sc]),
                             "producer_bias": float(source_b[sc]),
                             "producer_weights_exactly_zero": bool(np.all(source_w[sc] == 0)),
                             "producer_activation_flag": source_r["activation_flag_candidate"],
                             "conditional_upper_bound_if_nonnegative_signed16_predecessor": upper,
                             "conditional_input_cap": cap,
                             "consumer_weight_rms": float(dst_rms[channel]),
                             "consumer_input_rms_rank": rank,
                             "consumer_weights_exactly_zero": bool(np.all(dst_w[:, channel] == 0))})
        edge_constraints.append({"producer_layer": origin, "producer_name": source_r["name"],
                                 "consumer_layer": r["index"], "consumer_name": r["name"],
                                 "serialized_consumer_edge": edge, "resolved_channel_mapping": mapping,
                                 "selected_channels": selected,
                                 "bound_limit": "Upper bound assumes ReLU and signed16 positive cap; these runtime semantics are not verified."})

    # Raw zero indices are independent of convolution permutation code.
    zero_witnesses = []
    for name in ("conv3", "inc3e/conv3_2"):
        r = next(x for x in params["convolutions"] if x["name"] == name)
        raw = tensors[r["tensor_key"] + "_weight_i16_packed"].reshape(-1)
        outputs, inputs = weights[r["index"]].shape[:2]
        block_size = inputs * 9 * 4
        common = np.all(raw.reshape(outputs // 4, block_size) == 0, axis=0)
        predicted = {}
        for c in np.flatnonzero(np.all(weights[r["index"]] == 0, axis=(0, 2, 3))):
            offsets = sorted((k // 4) * 16 + o * 4 + k % 4
                             for k in range(int(c) * 9, (int(c) + 1) * 9) for o in range(4))
            require(np.all(common[offsets]), "Zero witness does not match original byte offsets")
            predicted[str(int(c))] = offsets
        zero_witnesses.append({"name": name, "weight_data_offset": r["weight"]["data_offset"],
                               "O4_block_elements": block_size,
                               "common_zero_i16_offsets_in_each_O4_block": np.flatnonzero(common).tolist(),
                               "K4_IHW_zero_channel_offsets": predicted,
                               "first_O4_block": bytes_evidence(data, r["weight"]["data_offset"], block_size * 2)})

    input_array = tensors["embedded_input_i16"]
    width = params["input_candidate"]["valid_width"]
    aux = params["input_auxiliary"]
    bins = list(struct.unpack_from("<16I", data, aux["offset"]))
    require(sum(bins) == 256 * 384, "Auxiliary histogram count changed")
    input_evidence = {"descriptor": params["input_candidate"],
                      "mean_like_blob_sha256": digest(input_array.tobytes()),
                      "valid_area_sha256": digest(input_array[..., :width].tobytes()),
                      "two_tail_elements_per_row": {
                          "sha256": digest(input_array[..., width:].tobytes()),
                          "unique_i16_values": int(np.unique(input_array[..., width:]).size),
                          "all_zero": bool(np.all(input_array[..., width:] == 0)),
                          "first16_rows_i16": input_array[0, 0, :16, width:].tolist()},
                      "auxiliary": bytes_evidence(data, aux["offset"], aux["bytes"]),
                      "auxiliary_uint32_counts": bins,
                      "counts_sum": sum(bins),
                      "limit": "A constant117 tensor and histogram-like auxiliary bytes do not prove runtime subtraction, channel format or histogram processing."}
    misc = []
    for n in graph["layers"]:
        if n["operator_id"] not in (5, 9, 10):
            continue
        q = tail_offset(n)
        misc.append({"layer": n["index"], "name": n["name"], "operator_id": n["operator_id"],
                     "inputs": n["inputs"], "metadata_candidate": analysis[n["index"]],
                     "record_tail": bytes_evidence(data, q, n["record_end"] - q)})
    return {"source_sha256": digest(data), "input_evidence": input_evidence,
            "bias_layout": bias_rows, "classification_bias_lane_constraints": heads,
            "weight_tile_constraints": rows, "cross_layer_sparsity_constraints": edge_constraints,
            "original_byte_zero_witnesses": zero_witnesses, "pool_concat_slice_records": misc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument("--output", type=Path, default=BASE / "artifacts/recognition-diagnosis-v3/static-constraints")
    args = parser.parse_args()
    source = resolve_source(args)
    args.output.mkdir(parents=True, exist_ok=True)
    models = {}
    for stem, expected in MODELS.items():
        data = (source / (stem + ".bin")).read_bytes()
        require(digest(data) == expected, "Original model identity mismatch")
        models[stem] = inspect_model(data)
    source_names = [Path(__file__).name, "decode_parameters.py", "extract.py", "ops_investigation.py", "packing_investigation.py"]
    report = {"schema_version": 1, "status": "static evidence supports a new flattened-K4 candidate; original DSP equivalence unverified",
              "scope": {"model_components": list(MODELS), "images_read": 0, "optimizer_steps": 0,
                        "AP_evaluations": 0, "core_files_modified": False},
              "source_hashes": {name: digest((BASE / name).read_bytes()) for name in source_names},
              "candidate": {"name": "gemm_K4_IHW", "physical_axes": ["O/4", "(I*H*W)/4", "O4", "K4"],
                            "flattened_K_order": "I,H,W", "formula": "raw.reshape(O//4,(I*H*W)//4,4,4).transpose(0,2,1,3).reshape(O,I,H,W)",
                            "scope": "ordinary convolutions including conv1 and1x1; upsample left unchanged and unresolved",
                            "why_new": "K4 tiles cross the nine-tap input-kernel boundary; no simple permutation of separately split I4,H,W axes represents this mapping."},
              "observations": [
                  "Bias int32 values are repeated as adjacent identical pairs; padded bias rows are zero.",
                  "Classification bias lane-order constraints independently match the already supported1x1 output order.",
                  "Flattened K4 is identical to previous H_O_W conv1 candidate and to existing1x1 packing.",
                  "Selected3x3 original-byte zero patterns become complete input channels under K4-IHW and align with upstream zero/near-zero weights and negative biases through serialized slice ports.",
                  "Input valid area is constant117 in both models and16-count auxiliary is identical; the two extra stored elements per row are nonzero and differ between models, with unknown purpose."],
              "inferences": [
                  "The K4-IHW packing candidate is better constrained by cross-layer sparsity than a smoothness or AP choice.",
                  "Existing adjacent-pair bias channel order and selected slice port mapping receive independent support; there is no current evidence justifying a bias permutation.",
                  "ReLU semantics are consistent with pruning zero/near-zero negative-bias outputs, but are not established by static flags alone."],
              "unknowns": ["Original XM4 numerical output and kernel addressing are unavailable.",
                           "No camera preprocessing, pool max-vs-average, fixed-point saturation/rounding or complete upsample routing is confirmed.",
                           "Static sparsity correlation is not proof that this candidate restores detection; a separately frozen train-only probe is needed."],
              "models": models}
    out = args.output / "static-constraints.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"report": str(out), "report_sha256": digest(out.read_bytes()),
                      "script_sha256": report["source_hashes"][Path(__file__).name],
                      "candidate": report["candidate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
