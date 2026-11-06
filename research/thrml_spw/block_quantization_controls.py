"""Explicit stronger low-bit controls; no label-based tuning or native sampling.

Preserving zero and localizing input groups can matter for pretrained weights.
Report every predeclared variant, rather than treating one poor affine lattice
as evidence that all conventional INT4 methods fail.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def quantize(weight, bits, method, block_size=None):
    weight = np.asarray(weight, dtype=np.float32)
    if block_size is not None:
        if weight.shape[0] % block_size:raise ValueError("Input dimension must divide group size")
        grouped = weight.reshape(-1, block_size, weight.shape[1]);axis = 1
    else:grouped, axis = weight, 0
    if method == "symmetric":
        limit = (1 << (bits-1))-1
        scale = np.max(np.abs(grouped), axis=axis, keepdims=True)/limit
        code = np.zeros_like(grouped);np.divide(grouped, scale, out=code, where=scale != 0)
        result = np.clip(np.rint(code), -limit, limit)*scale
    elif method == "affine_zero":
        lower = np.minimum(grouped.min(axis=axis, keepdims=True), 0)
        upper = np.maximum(grouped.max(axis=axis, keepdims=True), 0)
        limit = (1 << bits)-1;scale = (upper-lower)/limit
        zero = np.zeros_like(lower);np.divide(-lower, scale, out=zero, where=scale != 0)
        zero = np.clip(np.rint(zero), 0, limit)
        code = np.zeros_like(grouped);np.divide(grouped, scale, out=code, where=scale != 0)
        result = (np.clip(np.rint(code)+zero, 0, limit)-zero)*scale
    else:raise ValueError("Unknown quantizer")
    return result.reshape(weight.shape).astype(np.float32)


def main():
    import torch
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers import AutoModelForCausalLM
    from .transformer_block_gpu import BlockRunner
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    source = Path("results/thrml_spw/transformer_block0_gpu")
    protocol = json.loads((source/"protocol.json").read_text())
    out = Path("results/thrml_spw/block_quantization_controls");out.mkdir(parents=True, exist_ok=True)
    candidates = [("symmetric_output4",4,"symmetric",None), ("affine_zero_output4",4,"affine_zero",None),
                  ("symmetric_block128_4",4,"symmetric",128), ("affine_zero_block128_4",4,"affine_zero",128),
                  ("symmetric_output8",8,"symmetric",None), ("affine_zero_output8",8,"affine_zero",None)]
    design = {"candidates": candidates, "selection": "None; all variants reported. Weights-only scales, no label fitting.",
              "reason": "Follow up poor floating-offset affine4 baseline with zero-preserving and localized controls before any broad INT4 claim.",
              "symmetric_levels": "[-(2^(b-1)-1), +(2^(b-1)-1)], exact zero, one signed code unused",
              "affine_zero_levels": "[0,2^b-1], integer zero point, range includes zero",
              "group128": "128 input coefficients per output column, not128x128 tiles",
              "scope": "Numerical dequantized inference, not an optimized packed low-bit kernel; no best-available GPTQ/AWQ claim",
              "model_revision": protocol["revision"], "targets": protocol["targets_in_forward_order"],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"protocol.json").write_text(json.dumps(design,indent=2)+"\n")
    snapshot = Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['revision']}"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager").float().eval()
    runner = BlockRunner(model)
    data = np.load("data/thrml_spw/transformer_projection/export.npz")
    rows, raw = [], {}
    for name, bits, method, block in candidates:
        try:
            with torch.no_grad():
                for module, w in zip(runner.modules, runner.weights):
                    q = quantize(w,bits,method,block)
                    stored = q if module.__class__.__name__ == "Conv1D" else q.T
                    module.weight.copy_(torch.from_numpy(stored))
            groups = sum(w.shape[1]*(w.shape[0]//block if block else 1) for w in runner.weights)
            for split in ("validation","test"):
                result = runner.evaluate(data[split+"_tokens"])
                rows.append({"split":split,"condition":name,"bits":bits,"method":method,"input_block_size":block or 0,
                    **result["scores"],"groups":groups,"nominal_code_bits":protocol["converted_matrix_weights"]*bits,
                    "nominal_scale_bits":groups*32,"nominal_zero_point_bits":groups*bits if method=="affine_zero" else 0})
                raw[f"{split}_{name}_nll"],raw[f"{split}_{name}_predictions"] = result["nll"],result["predictions"]
                print({"condition":name,"split":split,**result["scores"]},flush=True)
        finally:runner.restore()
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/"tokens.npz",**raw)
    (out/"manifest.json").write_text(json.dumps({"config":design,"results":rows,"torch_version":torch.__version__},indent=2)+"\n")


if __name__ == "__main__":main()
