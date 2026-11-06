"""Sequential native conversion of all four projections in DistilGPT-2 block0.

Run in .venv. A persistent .venv-thrml worker accumulates native linear outputs.
Later student inputs are regenerated after all earlier replacements. Raw arrays,
prefix token losses, input checks, stage signatures and atomic checkpoints persist.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM

from .transformer_export import token_metrics


TARGETS = ["transformer.h.0.attn.c_attn", "transformer.h.0.attn.c_proj",
           "transformer.h.0.mlp.c_fc", "transformer.h.0.mlp.c_proj"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):h.update(block)
    return h.hexdigest()


def save_array(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as f:np.save(f, np.asarray(value))
    os.replace(temporary, path)


def save_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n");os.replace(temporary, path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def score(nll, predictions, tokens):
    ce = float(nll.astype(np.float64).mean())
    return {"cross_entropy": ce, "perplexity": float(np.exp(ce)),
            "token_accuracy": float(np.mean(predictions == tokens[:, 1:]))}


class BlockRunner:
    def __init__(self, model):
        self.model = model
        self.modules = [model.get_submodule(name) for name in TARGETS]
        self.forwards = [module.forward for module in self.modules]
        self.original_weights = [m.weight.detach().clone() for m in self.modules]
        self.weights = [m.weight.detach().numpy().copy() if m.__class__.__name__ == "Conv1D"
                        else m.weight.detach().numpy().T.copy() for m in self.modules]
        self.biases = [m.bias.detach().numpy().copy() for m in self.modules]

    def restore(self):
        with torch.no_grad():
            for m, w in zip(self.modules, self.original_weights):m.weight.copy_(w)
        for m, fn in zip(self.modules, self.forwards):m.forward = fn

    def evaluate(self, tokens, replacements=None, capture=None):
        replacements = replacements or {}
        nlls, predictions, inputs, outputs = [], [], [], []
        input_checks = 0
        hook = None
        if capture is not None:
            def collect(module, args, output):
                inputs.append(args[0].detach().numpy().copy())
                outputs.append(output.detach().numpy().copy())
            hook = self.modules[capture].register_forward_hook(collect)
        try:
            with torch.inference_mode():
                for case, token_row in enumerate(tokens):
                    for j, module in enumerate(self.modules):
                        if j not in replacements:
                            module.forward = self.forwards[j]
                            continue
                        cached_x, cached_y = replacements[j]
                        def replace(x, *args, _j=j, _case=case, **kwargs):
                            nonlocal input_checks
                            source_x, source_y = replacements[_j]
                            np.testing.assert_array_equal(x.detach().numpy(), source_x[_case:_case+1])
                            input_checks += 1
                            return torch.from_numpy(np.asarray(source_y[_case:_case+1]).copy()).to(dtype=x.dtype, device=x.device)
                        module.forward = replace
                    tensor = torch.from_numpy(token_row[None])
                    logits = self.model(input_ids=tensor, use_cache=False).logits
                    nll, predicted = token_metrics(logits, tensor)
                    nlls.append(nll[0]);predictions.append(predicted[0])
        finally:
            if hook is not None:hook.remove()
            for m, fn in zip(self.modules, self.forwards):m.forward = fn
        nll, pred = np.stack(nlls), np.stack(predictions)
        return {"nll": nll, "predictions": pred, "scores": score(nll, pred, tokens),
                "inputs": np.concatenate(inputs) if inputs else None,
                "outputs": np.concatenate(outputs) if outputs else None, "input_checks": input_checks}

    def quantize(self, bits):
        with torch.no_grad():
            for m, w in zip(self.modules, self.weights):
                if bits == 0:q = np.zeros_like(w)
                else:
                    lower, span = w.min(0), w.max(0)-w.min(0)
                    p = np.zeros_like(w);np.divide(w-lower, span, out=p, where=span != 0)
                    q = lower+span*np.rint(p*((1 << bits)-1))/((1 << bits)-1)
                stored = q if m.__class__.__name__ == "Conv1D" else q.T
                m.weight.copy_(torch.from_numpy(stored))


def main():
    if "--probe-worker" in sys.argv:
        # Keep the venv executable path: resolving its symlink invokes the base
        # interpreter and silently loses the native environment's site packages.
        probe = subprocess.Popen([str(Path('.venv-thrml/bin/python').absolute()), "-u", "-m", "research.thrml_spw.block_native_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
        try:
            line = probe.stdout.readline()
            if not line:raise RuntimeError("Native worker exited before ready")
            ready = json.loads(line);assert ready["status"] == "ready"
            probe.stdin.write('{"action":"quit"}\n');probe.stdin.flush()
            assert probe.wait(timeout=30) == 0
            print(json.dumps({"native_environment_probe": ready}), flush=True)
        finally:
            if probe.poll() is None:probe.terminate();probe.wait(timeout=30)
            probe.stdin.close();probe.stdout.close()
        return
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    root = Path("results/thrml_spw")
    out = root/"transformer_block0";out.mkdir(parents=True, exist_ok=True)
    data_path = Path("data/thrml_spw/transformer_projection/export.npz")
    data = np.load(data_path)
    prepared = json.loads((root/"transformer_preparation/manifest.json").read_text())
    assert sha(data_path) == prepared["export_sha256"]
    snapshot = Path(HF_HUB_CACHE)/f"models--distilbert--distilgpt2/snapshots/{prepared['config']['model_revision']}"
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, attn_implementation="eager").float().eval()
    runner = BlockRunner(model)
    matrix_parameters = sum(w.size for w in runner.weights)
    protocol = {"model": prepared["config"]["model"], "revision": prepared["config"]["model_revision"],
        "targets_in_forward_order": TARGETS, "converted_matrix_weights": int(matrix_parameters),
        "model_parameters": prepared["total_parameters"], "converted_fraction": matrix_parameters/prepared["total_parameters"],
        "matrix_shapes": [list(w.shape) for w in runner.weights], "deterministic_converted_module_biases": sum(b.size for b in runner.biases),
        "calibration_sample_search": [64, 256, 1024], "sampling_seeds": list(range(5)),
        "contexts_per_weight_bank_batch": 8, "calibration_domain": 940730, "test_domain": 950730,
        "selection": "In ascending T, run five calibration seeds; stop at first mean CE increase<=log(1.05) and mean token-accuracy drop<=0.01 versus FP32. Otherwise largest T is a failed-gate diagnostic. Test selected T and the predeclared initial64 without further selection.",
        "test_labels_used_in_selection": False, "input_export_sha256": sha(data_path),
        "input_regeneration": "Before each projection, rerun real model with every earlier native output substituted, assert every prefix input exactly, export current student input and conditional deterministic output",
        "sampling": "Official THRML per-output affine native linear-output accumulation before all downstream nonlinearities. Biases, LayerNorm, embeddings, output head and all other blocks deterministic.",
        "resume": "Atomic native artifacts reused only with matching input/reference/weight/source/key signatures and output checksums. Regenerate and recheck all student inputs on resume.",
        "scope": "One complete block, not whole-model conversion; short-context corpus subset; float32 controls and external arithmetic; no physical-system advantage claim",
        "driver_source_sha256": sha(__file__)}
    save_json(out/"protocol.json", protocol)
    master = out/"weights.npz"
    arrays = {**{f"weight_{i}": w for i, w in enumerate(runner.weights)}, **{f"bias_{i}": b for i, b in enumerate(runner.biases)}}
    if master.exists():
        old = np.load(master)
        for name, array in arrays.items():np.testing.assert_array_equal(old[name], array)
    else:np.savez(master, **arrays)
    started = time.monotonic();baselines = {};teacher = {}
    baseline_rows = [];baseline_raw = {}
    for split in ("validation", "test"):
        tokens = data[split+"_tokens"]
        teacher[split] = {}
        for stage in range(4):
            result = runner.evaluate(tokens, capture=stage)
            teacher[split][stage] = (result["inputs"], result["outputs"])
            save_array(out/"teacher"/f"{split}_stage{stage}_inputs.npy", result["inputs"])
            save_array(out/"teacher"/f"{split}_stage{stage}_outputs.npy", result["outputs"])
            expected = result["inputs"]@runner.weights[stage]+runner.biases[stage]
            error = float(np.max(np.abs(expected-result["outputs"])))
            assert error/(1+float(np.max(np.abs(result["outputs"])))) < 2e-5
        full = runner.evaluate(tokens, teacher[split])
        np.testing.assert_array_equal(full["nll"], data[split+"_reference_token_nll"])
        np.testing.assert_array_equal(full["predictions"], data[split+"_reference_predicted_tokens"])
        baselines[split] = full["scores"]
        baseline_rows.append({"split": split, "condition": "FP32", **full["scores"]})
        baseline_raw[split+"_FP32_nll"], baseline_raw[split+"_FP32_predictions"] = full["nll"], full["predictions"]
        for bits, label in ((0, "zero_matrices"), (4, "affine4"), (8, "affine8")):
            try:
                runner.quantize(bits);result = runner.evaluate(tokens)
                baseline_rows.append({"split": split, "condition": label, **result["scores"]})
                baseline_raw[f"{split}_{label}_nll"], baseline_raw[f"{split}_{label}_predictions"] = result["nll"], result["predictions"]
            finally:runner.restore()
        print(f"Block baselines/replay verified split={split} elapsed={time.monotonic()-started:.1f}s", flush=True)
    write_csv(out/"baseline_metrics.csv", baseline_rows)
    np.savez_compressed(out/"baseline_tokens.npz", **baseline_raw)
    worker_log = (out/"worker.log").open("a")
    worker = subprocess.Popen([str(Path('.venv-thrml/bin/python').absolute()), "-u", "-m", "research.thrml_spw.block_native_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True, bufsize=1,
        env={**os.environ, "JAX_PLATFORMS": "cpu"})
    rows = [];prefix_rows = [];run_records = []
    try:
        def response_from_worker():
            line = worker.stdout.readline()
            if not line:raise RuntimeError(f"Native worker closed stdout (exit={worker.poll()}); see {out/'worker.log'}")
            return json.loads(line)
        ready = response_from_worker();assert ready["status"] == "ready"
        def run(split, t, seed):
            directory = out/"runs"/f"{split}_T{t}_seed{seed}";directory.mkdir(parents=True, exist_ok=True)
            tokens = data[split+"_tokens"];replacements = {};raw = {};stages = []
            checks = 0
            for stage in range(4):
                print(f"Block native start split={split} T={t} seed={seed} stage={stage} elapsed={time.monotonic()-started:.1f}s", flush=True)
                capture = runner.evaluate(tokens, replacements, capture=stage)
                checks += capture["input_checks"]
                raw[f"prefix{stage}_nll"], raw[f"prefix{stage}_predictions"] = capture["nll"], capture["predictions"]
                prefix_rows.append({"split": split, "samples": t, "sampling_seed": seed, "converted_prefix": stage, **capture["scores"]})
                xp, rp, yp = [directory/f"stage{stage}_{name}.npy" for name in ("inputs", "conditional_reference", "native_output")]
                save_array(xp, capture["inputs"]);save_array(rp, capture["outputs"])
                request = {"weights": str(master), "inputs": str(xp), "reference": str(rp), "output": str(yp),
                    "stage": stage, "samples": t, "seed": seed,
                    "domain": protocol["calibration_domain"] if split == "validation" else protocol["test_domain"], "contexts_per_batch": 8}
                worker.stdin.write(json.dumps(request)+"\n");worker.stdin.flush()
                response = response_from_worker()
                if response["status"] != "ok":raise RuntimeError(f"Native worker failed: {response}")
                metadata = json.loads(Path(response["metadata"]).read_text())
                assert metadata["status"] == "complete" and metadata["output_sha256"] == sha(yp)
                y = np.load(yp, mmap_mode="r")
                replacements[stage] = (np.load(xp, mmap_mode="r"), y)
                stages.append({"stage": stage, "inputs": str(xp), "reference": str(rp), "output": str(yp),
                    "metadata": response["metadata"], "cached": response["cached"],
                    "student_input_rmse_vs_teacher": float(np.sqrt(np.mean((capture["inputs"].astype(np.float64)-teacher[split][stage][0])**2))),
                    "prior_native_output_sha256": [sha(directory/f"stage{k}_native_output.npy") for k in range(stage)]})
                print(f"Block native done split={split} T={t} seed={seed} stage={stage} cached={response['cached']} elapsed={time.monotonic()-started:.1f}s", flush=True)
            final = runner.evaluate(tokens, replacements);checks += final["input_checks"]
            raw["prefix4_nll"], raw["prefix4_predictions"] = final["nll"], final["predictions"]
            prefix_rows.append({"split": split, "samples": t, "sampling_seed": seed, "converted_prefix": 4, **final["scores"]})
            assert checks == len(tokens)*10
            np.savez_compressed(directory/"prefix_tokens.npz", **raw)
            record = {"status": "complete", "split": split, "samples": t, "sampling_seed": seed,
                      "scores": final["scores"], "stages": stages, "exact_prefix_input_checks": checks,
                      "free_node_updates": sum(json.loads(Path(s["metadata"]).read_text())["free_node_updates"] for s in stages),
                      "input_export_sha256": protocol["input_export_sha256"]}
            save_json(directory/"run.json", record);run_records.append(record)
            rows.append({"split": split, "samples": t, "sampling_seed": seed, **final["scores"],
                         "free_node_updates": record["free_node_updates"], "exact_prefix_input_checks": checks})
            write_csv(out/"native_metrics.csv", rows);write_csv(out/"prefix_metrics.csv", prefix_rows)
            print(f"Block complete split={split} T={t} seed={seed} ppl={final['scores']['perplexity']:.3f}", flush=True)
        profiles = [];selected = None
        for t in protocol["calibration_sample_search"]:
            for seed in range(5):run("validation", t, seed)
            group = [r for r in rows if r["split"] == "validation" and r["samples"] == t]
            ce = float(np.mean([r["cross_entropy"] for r in group]));acc = float(np.mean([r["token_accuracy"] for r in group]))
            passed = bool(ce-baselines["validation"]["cross_entropy"] <= np.log(1.05) and baselines["validation"]["token_accuracy"]-acc <= .01)
            profile = {"samples": t, "cross_entropy_mean": ce, "token_accuracy_mean": acc,
                       "relative_perplexity": float(np.exp(ce-baselines["validation"]["cross_entropy"])), "passes": passed}
            profiles.append(profile)
            save_json(out/"calibration_progress.json", {"profiles": profiles, "test_labels_used_in_selection": False})
            print("Block calibration", profile, flush=True)
            if passed:selected = profile;break
        if selected is None:selected = profiles[-1]
        selection = {"gate_passed_on_calibration": any(r["passes"] for r in profiles), "selected": selected,
                     "profiles": profiles, "test_labels_used_in_selection": False}
        save_json(out/"selection.json", selection)
        for t in sorted({64, selected["samples"]}):
            for seed in range(5):run("test", t, seed)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"action":"quit"}\n');worker.stdin.flush();worker.wait(timeout=60)
        worker.stdin.close();worker.stdout.close();worker_log.close()
    summary = []
    for split, t in sorted({(r["split"], r["samples"]) for r in rows}):
        group = [r for r in rows if r["split"] == split and r["samples"] == t]
        ce = np.array([r["cross_entropy"] for r in group]);mean = float(ce.mean());margin = float(1.96*ce.std(ddof=1)/np.sqrt(len(ce)))
        ref = baselines[split]["cross_entropy"]
        summary.append({"split": split, "samples": t, "sampling_seeds": len(group), "cross_entropy_mean": mean,
            "cross_entropy_ci95_margin": margin, "exp_mean_cross_entropy": float(np.exp(mean)),
            "relative_perplexity": float(np.exp(mean-ref)), "relative_perplexity_lower": float(np.exp(mean-margin-ref)),
            "relative_perplexity_upper": float(np.exp(mean+margin-ref)), "token_accuracy_mean": float(np.mean([r["token_accuracy"] for r in group])),
            "free_node_updates": int(group[0]["free_node_updates"]), "sequential_sampling_rounds": 4*t})
    write_csv(out/"summary.csv", summary)
    save_json(out/"manifest.json", {"config": protocol, "selection": selection, "summary": summary,
        "run_count": len(run_records), "elapsed_seconds": time.monotonic()-started,
        "all_student_inputs_regenerated_and_checked": True, "FP32_whole_block_replay_exact": True,
        "torch_version": torch.__version__, "weights_sha256": sha(master), "driver_source_sha256": sha(__file__),
        "scope": protocol["scope"]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":main()
