"""Replay actual native projection outputs through a frozen pretrained model.

Run in .venv (Torch), separately from .venv-thrml. Exact cached-input checks prove
that the exported operator is at the right place in the real forward path.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


def write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    root = Path("results/thrml_spw")
    out = root/"transformer_replay"
    out.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((root/"transformer_preparation/manifest.json").read_text())
    native_manifest = json.loads((root/"transformer_projection_native/manifest.json").read_text())
    data = np.load("data/thrml_spw/transformer_projection/export.npz")
    native_path = root/"transformer_projection_native/outputs.npz"
    native = np.load(native_path)
    protocol = {"model": prepared["config"]["model"], "revision": prepared["config"]["model_revision"],
                "target": prepared["config"]["target_module"], "converted_fraction": prepared["converted_fraction"],
                "converted_matrix_weights": prepared["converted_matrix_parameters"],
                "selection_rule": "Smallest T with mean calibration CE increase <=log(1.05) and mean token accuracy drop<=0.01 versus FP32; otherwise largest T is a diagnostic failed-gate fallback. Freeze selection before evaluating test labels.",
                "test_labels_used_in_selection": False,
                "sampling_scope": "Five Monte Carlo seeds of one frozen pretrained model on fixed disjoint validation/test windows; CIs quantify sampling variation, not training or corpus-wide uncertainty",
                "pipeline": "THRML directly accumulated linear outputs, then Torch replay inserted these before every downstream residual/activation/attention operation. Only the selected projection is converted.",
                "native_outputs_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                "official_simulator_commit": native_manifest["source_commit"]}
    (out/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    snapshot = Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['revision']}"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager").float().eval()
    target = model.get_submodule(protocol["target"])
    original = target.forward
    cases = [(name, -1, 0) for name in ("FP32", "zero_matrix", "affine4", "affine8")]
    cases += [(f"T{t}", seed, t) for t in (16, 64, 256) for seed in range(5)]
    rows, raw, start = [], {}, time.monotonic()
    selections = None
    for split in ("validation", "test"):
        ids, cached_inputs = data[split+"_tokens"], data[split+"_inputs"]
        raw[split+"_target_tokens"] = ids[:, 1:]
        for condition, seed, t in cases:
            key = f"{split}_T{t}_seed{seed}" if t else f"{split}_{condition}"
            replacements = native[key]
            nlls, predictions = [], []
            with torch.inference_mode():
                for i, tokens in enumerate(ids):
                    def replace(x, *args, _i=i, **kwargs):
                        np.testing.assert_array_equal(x.detach().cpu().numpy(), cached_inputs[_i:_i+1])
                        return torch.from_numpy(replacements[_i:_i+1]).to(dtype=x.dtype, device=x.device)
                    target.forward = replace
                    tensor = torch.from_numpy(tokens[None])
                    logits = model(input_ids=tensor, use_cache=False).logits[:, :-1].contiguous()
                    labels = tensor[:, 1:].contiguous()
                    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
                    nlls.append(nll.cpu().numpy())
                    predictions.append(logits.argmax(-1).cpu().numpy()[0])
            target.forward = original
            nlls, predictions = np.stack(nlls), np.stack(predictions)
            if condition == "FP32":
                np.testing.assert_array_equal(nlls, data[split+"_reference_token_nll"])
                np.testing.assert_array_equal(predictions, data[split+"_reference_predicted_tokens"])
            ce = float(nlls.astype(np.float64).mean())
            rows.append({"split": split, "condition": condition, "sampling_seed": seed, "samples": t,
                         "target_tokens": nlls.size, "cross_entropy": ce, "perplexity": float(np.exp(ce)),
                         "token_accuracy": float(np.mean(predictions == ids[:, 1:]))})
            raw[key+"_token_nll"], raw[key+"_predicted_tokens"] = nlls, predictions
        print(f"Transformer replay split={split} elapsed={time.monotonic()-start:.1f}s", flush=True)
        if split == "validation":
            baseline = next(r for r in rows if r["split"] == split and r["condition"] == "FP32")
            profiles = []
            for t in (16, 64, 256):
                group = [r for r in rows if r["split"] == split and r["samples"] == t]
                ce, acc = float(np.mean([r["cross_entropy"] for r in group])), float(np.mean([r["token_accuracy"] for r in group]))
                profiles.append({"samples": t, "mean_calibration_ce": ce, "mean_calibration_token_accuracy": acc,
                    "relative_perplexity": float(np.exp(ce-baseline["cross_entropy"])),
                    "passes": bool(ce-baseline["cross_entropy"] <= np.log(1.05) and baseline["token_accuracy"]-acc <= .01)})
            passing = [r for r in profiles if r["passes"]]
            selected = min(passing, key=lambda r: r["samples"]) if passing else profiles[-1]
            selections = {"gate_passed_on_calibration": bool(passing), "selected": selected,
                          "profiles": profiles, "test_labels_used_in_selection": False}
            (out/"selection.json").write_text(json.dumps(selections, indent=2)+"\n")
            print("Transformer selection frozen:", selected, flush=True)
    write(out/"metrics.csv", rows)
    np.savez_compressed(out/"token_evidence.npz", **raw)
    summary = []
    for split in ("validation", "test"):
        baseline = next(r for r in rows if r["split"] == split and r["condition"] == "FP32")
        for condition in dict.fromkeys(r["condition"] for r in rows if r["split"] == split):
            group = [r for r in rows if r["split"] == split and r["condition"] == condition]
            ce = np.array([r["cross_entropy"] for r in group])
            mean = float(ce.mean())
            margin = float(1.96*ce.std(ddof=1)/np.sqrt(len(ce))) if len(ce) > 1 else 0.
            summary.append({"split": split, "condition": condition, "sampling_seeds": len(group),
                "cross_entropy_mean": mean, "cross_entropy_ci95_margin": margin,
                "exp_mean_cross_entropy": float(np.exp(mean)),
                "relative_perplexity_from_mean_ce": float(np.exp(mean-baseline["cross_entropy"])),
                "relative_perplexity_ci95_lower": float(np.exp(mean-margin-baseline["cross_entropy"])),
                "relative_perplexity_ci95_upper": float(np.exp(mean+margin-baseline["cross_entropy"])),
                "token_accuracy_mean": float(np.mean([r["token_accuracy"] for r in group]))})
    write(out/"summary.csv", summary)
    result = {"config": protocol, "selection": selections, "summary": summary,
              "all_cached_inputs_exactly_matched": True, "FP32_token_replay_exact": True,
              "torch_version": torch.__version__, "elapsed_seconds": time.monotonic()-start,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Real pretrained downstream computation on a short-context subset, one projection converted. Not full-corpus benchmark or full-model probabilistic execution."}
    (out/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
