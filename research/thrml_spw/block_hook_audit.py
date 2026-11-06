"""Independent forward-hook replay of completed multi-operator checkpoints.

Unlike the experiment driver, this checker leaves each module's forward method
intact, verifies inputs in pre-hooks and replaces outputs in post-hooks. It checks
all saved completed runs against the real frozen model and raw token losses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/thrml_spw/transformer_block0_gpu")
    parser.add_argument("--tokens", help="Optional reserved token archive; uses its tokens key for every run")
    args = parser.parse_args()
    out = Path(args.root)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    protocol = json.loads((out/"protocol.json").read_text())
    data = np.load(args.tokens or "data/thrml_spw/transformer_projection/export.npz")
    snapshot = Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{protocol['revision']}"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager").float().eval()
    modules = [model.get_submodule(name) for name in protocol["targets_in_forward_order"]]
    results = []
    for run_path in sorted((out/"runs").glob("*/run.json")):
        record = json.loads(run_path.read_text())
        if record["status"] != "complete":continue
        split = record["split"];tokens = data["tokens" if args.tokens else split+"_tokens"]
        inputs = [np.load(s["inputs"], mmap_mode="r") for s in record["stages"]]
        outputs = [np.load(s["output"], mmap_mode="r") for s in record["stages"]]
        raw = np.load(run_path.parent/"prefix_tokens.npz")
        case = 0;checks = 0;hooks = []
        for stage, module in enumerate(modules):
            def before(module, args, _stage=stage):
                nonlocal checks
                np.testing.assert_array_equal(args[0].detach().numpy(), inputs[_stage][case:case+1])
                checks += 1
            def after(module, args, result, _stage=stage):
                return torch.from_numpy(np.asarray(outputs[_stage][case:case+1]).copy()).to(dtype=result.dtype, device=result.device)
            hooks.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
        nlls, predictions = [], []
        try:
            with torch.inference_mode():
                for case, token_row in enumerate(tokens):
                    ids = torch.from_numpy(token_row[None])
                    logits = model(input_ids=ids, use_cache=False).logits[:, :-1].contiguous()
                    target = ids[:, 1:].contiguous()
                    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
                    nlls.append(nll.numpy());predictions.append(logits.argmax(-1).numpy()[0])
        finally:
            for hook in hooks:hook.remove()
        np.testing.assert_array_equal(np.stack(nlls), raw["prefix4_nll"])
        np.testing.assert_array_equal(np.stack(predictions), raw["prefix4_predictions"])
        assert checks == 4*len(tokens)
        results.append({"run": str(run_path), "exact_native_prefix_inputs": checks,
                        "final_token_losses_exact": True, "final_token_predictions_exact": True})
        print(f"Independent hooks verified {run_path.parent.name}", flush=True)
    final = {"passed": bool(results), "completed_runs_audited": len(results), "runs": results,
             "scope": "Independent pre/post-hook execution checks on all currently completed runs; final completion still requires planned calibration/test run counts."}
    (out/"independent_hook_audit.json").write_text(json.dumps(final, indent=2)+"\n")
    print(json.dumps({k:v for k,v in final.items() if k != "runs"}), flush=True)


if __name__ == "__main__":main()
