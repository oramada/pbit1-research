"""Prepare a useful pretrained Transformer for a native single-operator study.

Torch and THRML remain in separate environments. Export a frozen projection's
inputs/outputs; native output replacements can later be replayed through the rest
of the unchanged model. This module verifies exact FP32 replay before conversion.
It does not itself demonstrate stochastic Transformer inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"
TARGET = "transformer.h.0.attn.c_proj"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def windows(tokenizer, path, count, seed, sequence_length=64):
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    text = "\n".join(s for s in texts if s and s.strip())
    ids = np.array(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    width = sequence_length+1
    blocks = len(ids)//width
    selected = np.sort(np.random.default_rng(seed).choice(blocks, count, replace=False))
    return np.stack([ids[b*width:(b+1)*width] for b in selected]), selected*width


def token_metrics(logits, ids):
    pred, target = logits[:, :-1].contiguous(), ids[:, 1:].contiguous()
    nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="none")
    return nll.reshape(target.shape).cpu().numpy(), pred.argmax(-1).cpu().numpy()


def main():
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    out, evidence = Path("data/thrml_spw/transformer_projection"), Path("results/thrml_spw/transformer_preparation")
    out.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    protocol = {"model": "distilbert/distilgpt2", "model_revision": MODEL_REVISION,
                "target_module": TARGET, "selection": "Predeclared first attention output projection; no task-based component selection",
                "calibration_windows": 16, "test_windows": 32, "sequence_length": 64,
                "data": "Separate official WikiText-2 raw validation/test parquet splits; pinned dataset revision, disjoint nonoverlapping token tiles within each split",
                "scope": "Preparation and exact FP32 output-replay check only. Later native conversion is selective, not a full-model one-p-bit claim.",
                "references": [f"https://huggingface.co/distilbert/distilgpt2/tree/{MODEL_REVISION}",
                               "https://huggingface.co/datasets/Salesforce/wikitext"]}
    (evidence/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    snapshot = Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{MODEL_REVISION}"
    # The model is already cached; the explicit snapshot prevents revision drift.
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager").float().eval()
    target = model.get_submodule(TARGET)
    if isinstance(target, torch.nn.Linear):
        weight = target.weight.detach().numpy().T.copy()
    elif target.__class__.__name__ == "Conv1D":
        weight = target.weight.detach().numpy().copy()
    else:
        raise TypeError(f"Unsupported projection type: {type(target)}")
    bias = target.bias.detach().numpy().copy()
    dataset_revision = HfApi(token=False).dataset_info("Salesforce/wikitext").sha
    paths = {split: Path(hf_hub_download("Salesforce/wikitext",
        f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset", revision=dataset_revision, token=False))
        for split in ("validation", "test")}
    arrays = {"weight": weight, "bias": bias}
    report, start, errors = {}, time.monotonic(), []
    for split, count, seed in (("validation", 16, 20261030), ("test", 32, 20261031)):
        ids, offsets = windows(tokenizer, paths[split], count, seed)
        captured_in, captured_out, nlls, predictions = [], [], [], []
        def capture(module, args, output):
            captured_in.append(args[0].detach().cpu().numpy().copy())
            captured_out.append(output.detach().cpu().numpy().copy())
        hook = target.register_forward_hook(capture)
        with torch.inference_mode():
            for tokens in ids:
                tensor = torch.from_numpy(tokens[None])
                logits = model(input_ids=tensor, use_cache=False).logits
                nll, prediction = token_metrics(logits, tensor)
                nlls.append(nll[0]);predictions.append(prediction[0])
        hook.remove()
        inputs, outputs = np.concatenate(captured_in), np.concatenate(captured_out)
        projection_error = float(np.max(np.abs(inputs@weight+bias-outputs)))
        if projection_error > 1e-4:
            raise RuntimeError(f"Weight orientation/export mismatch: {projection_error}")
        original_forward = target.forward
        replay_nll, replay_prediction = [], []
        with torch.inference_mode():
            for i, tokens in enumerate(ids):
                def replace(x, *args, _i=i, **kwargs):
                    np.testing.assert_array_equal(x.detach().cpu().numpy(), inputs[_i:_i+1])
                    return torch.from_numpy(outputs[_i:_i+1]).to(dtype=x.dtype, device=x.device)
                target.forward = replace
                tensor = torch.from_numpy(tokens[None])
                logits = model(input_ids=tensor, use_cache=False).logits
                nll, prediction = token_metrics(logits, tensor)
                replay_nll.append(nll[0]);replay_prediction.append(prediction[0])
        target.forward = original_forward
        np.testing.assert_array_equal(np.stack(replay_nll), np.stack(nlls))
        np.testing.assert_array_equal(np.stack(replay_prediction), np.stack(predictions))
        arrays.update({f"{split}_tokens": ids, f"{split}_token_offsets": offsets,
                       f"{split}_inputs": inputs, f"{split}_reference_outputs": outputs,
                       f"{split}_reference_token_nll": np.stack(nlls),
                       f"{split}_reference_predicted_tokens": np.stack(predictions)})
        ce = float(np.mean(nlls))
        report[split] = {"windows": count, "target_tokens": count*64, "cross_entropy": ce,
                         "perplexity": float(np.exp(ce)),
                         "token_accuracy": float(np.mean(np.stack(predictions) == ids[:, 1:])),
                         "projection_export_max_error": projection_error, "exact_FP32_replay": True}
        errors.append(projection_error)
        print({"split": split, **report[split], "elapsed_seconds": time.monotonic()-start}, flush=True)
    np.savez_compressed(out/"export.npz", **arrays)
    manifest = {"config": protocol, "model_snapshot_path": str(snapshot), "dataset_revision": dataset_revision,
                "dataset_sha256": {k: digest(p) for k, p in paths.items()},
                "model_file_sha256": {p.name: digest(p) for p in snapshot.iterdir() if p.is_file()},
                "torch_version": torch.__version__, "target_type": target.__class__.__name__,
                "total_parameters": sum(p.numel() for p in model.parameters()), "converted_matrix_parameters": weight.size,
                "weight_shape_input_output": list(weight.shape), "converted_fraction": weight.size/sum(p.numel() for p in model.parameters()),
                "results": report, "elapsed_seconds": time.monotonic()-start,
                "export_sha256": digest(out/"export.npz"), "source_sha256": digest(__file__),
                "next_step": "Native THRML per-output affine sample/accumulation, then replay native outputs before all downstream nonlinearities. No stochastic language-model result yet."}
    (evidence/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
