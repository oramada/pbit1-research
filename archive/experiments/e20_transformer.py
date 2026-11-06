"""Small pretrained Transformer conversion pilot (DistilGPT-2 + WikiText-2).

Only transformer projection matrices are converted initially. Embeddings, LayerNorm,
and the tied output head remain deterministic as laid out in the design.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from weight_encodings.single_pbit import SinglePBit


MODEL_ID = "distilbert/distilgpt2"
WIKITEXT_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)


def load_token_windows(tokenizer, parquet_path: Path, n_windows: int, seq_len: int,
                       seed: int) -> torch.Tensor:
    table = pq.read_table(parquet_path, columns=["text"])
    text = "\n".join(s for s in table.column("text").to_pylist() if s and s.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    max_start = max(0, len(ids) - seq_len - 1)
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(max_start + 1, size=min(n_windows, max_start + 1),
                                replace=False))
    return torch.stack([ids[int(i):int(i) + seq_len + 1] for i in starts])


def selected_projection_modules(model):
    result = []
    for name, module in model.named_modules():
        if not name.startswith("transformer.h."):
            continue
        if getattr(module, "weight", None) is None or module.weight.ndim != 2:
            continue
        # GPT-2 uses Conv1D storage [input, output]; include only actual affine maps.
        if module.__class__.__name__ not in ("Conv1D", "Linear"):
            continue
        result.append((name, module))
    return result


def logits_on(model, inputs, batch_size, device):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(inputs), batch_size):
            x = inputs[begin:begin + batch_size].to(device)
            outputs.append(model(input_ids=x, use_cache=False).logits.cpu())
    return torch.cat(outputs)


def metrics(logits, inputs):
    pred = logits[:, :-1, :].contiguous()
    target = inputs[:, 1:].contiguous()
    loss = F.cross_entropy(pred.view(-1, pred.shape[-1]), target.view(-1))
    token_acc = (pred.argmax(-1) == target).float().mean()
    return float(loss), float(torch.exp(loss)), float(token_acc)


def quantize_model(model, bits):
    out = copy.deepcopy(model)
    levels = (1 << (bits - 1)) - 1
    with torch.no_grad():
        for _, module in selected_projection_modules(out):
            scale = module.weight.abs().max() / levels
            if float(scale) > 0:
                module.weight.copy_(torch.round(module.weight / scale).clamp(-levels, levels) * scale)
    return out


def encode_spw(model):
    encoded = []
    metadata_bytes = 0
    n_params = 0
    for name, module in selected_projection_modules(model):
        w = module.weight.detach().cpu().numpy().astype(np.float64)
        # Conv1D: final axis is output feature; Linear: first axis is output feature.
        if module.__class__.__name__ == "Conv1D":
            ids = np.broadcast_to(np.arange(w.shape[1])[None, :], w.shape)
        else:
            ids = np.broadcast_to(np.arange(w.shape[0])[:, None], w.shape)
        codec = SinglePBit(eps=1e-6)
        state = codec.encode(w, ids)
        encoded.append((name, module, codec, state))
        metadata_bytes += state.n_groups * 8
        n_params += w.size
    return encoded, metadata_bytes, n_params


def set_weights(encoded, samples, rng):
    for _, module, codec, state in encoded:
        w = codec.sample(state, samples, rng).mean(axis=0)
        module.weight.data.copy_(torch.from_numpy(w).to(
            device=module.weight.device, dtype=module.weight.dtype))


def make_rows(model, tokenizer, inputs, args):
    device = torch.device("cpu")
    model.to(device).eval()
    rows = []
    fp = logits_on(model, inputs, args.batch_size, device)
    loss, ppl, token_acc = metrics(fp, inputs)
    base = {"model": MODEL_ID, "representation": "FP32", "samples": 0,
            "converted_parameters": 0, "converted_fraction": 0.0,
            "metadata_bytes_float32": 0, "cross_entropy": loss,
            "perplexity": ppl, "token_accuracy": token_acc, "logit_rmse": 0.0}
    rows.append(base)
    for bits in (8, 4):
        qmodel = quantize_model(model, bits)
        qlogits = logits_on(qmodel, inputs, args.batch_size, device)
        qloss, qppl, qacc = metrics(qlogits, inputs)
        rows.append({**base, "representation": f"INT{bits}", "samples": 0,
                     "cross_entropy": qloss, "perplexity": qppl,
                     "token_accuracy": qacc,
                     "logit_rmse": float(torch.mean((qlogits - fp) ** 2).sqrt())})
    encoded, metadata, nparams = encode_spw(model)
    original = [(module, module.weight.detach().clone()) for _, module, _, _ in encoded]
    print(f"converted {len(encoded)} projection matrices / {nparams:,} params; "
          f"metadata={metadata:,} bytes", flush=True)
    for samples in args.samples:
        rng = np.random.default_rng(args.seed)
        set_weights(encoded, samples, rng)
        reconstructed = logits_on(model, inputs, args.batch_size, device)
        rloss, rppl, racc = metrics(reconstructed, inputs)
        rows.append({**base, "representation": "SPW_reconstructed", "samples": samples,
                     "converted_parameters": nparams,
                     "converted_fraction": nparams / sum(p.numel() for p in model.parameters()),
                     "metadata_bytes_float32": metadata,
                     "cross_entropy": rloss, "perplexity": rppl,
                     "token_accuracy": racc,
                     "logit_rmse": float(torch.mean((reconstructed - fp) ** 2).sqrt())})
    # End-to-end stochastic execution: draw one complete set of matrix weights per
    # realization, evaluate, then average the predictive logits. Includes nonlinearities.
    for samples in args.samples:
        rng = np.random.default_rng(args.seed)
        total = None
        for _ in range(samples):
            set_weights(encoded, 1, rng)
            current = logits_on(model, inputs, args.batch_size, device)
            total = current if total is None else total + current
        average = total / samples
        rloss, rppl, racc = metrics(average, inputs)
        rows.append({**base, "representation": "SPW_direct_logit_average", "samples": samples,
                     "converted_parameters": nparams,
                     "converted_fraction": nparams / sum(p.numel() for p in model.parameters()),
                     "metadata_bytes_float32": metadata,
                     "cross_entropy": rloss, "perplexity": rppl,
                     "token_accuracy": racc,
                     "logit_rmse": float(torch.mean((average - fp) ** 2).sqrt())})
    for module, weight in original:
        module.weight.data.copy_(weight)
    return rows, {"fp_perplexity": ppl, "fp_token_accuracy": token_acc,
                  "converted_parameters": nparams, "total_parameters": sum(p.numel() for p in model.parameters())}


def main():
    global MODEL_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--parquet", default="data/wikitext/test.parquet")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--out", default="results/transformer/distilgpt2_wikitext2.csv")
    args = parser.parse_args()
    MODEL_ID = args.model
    torch.set_num_threads(args.threads)
    start = time.time()
    parquet = Path(args.parquet)
    if not parquet.exists():
        import urllib.request
        parquet.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WIKITEXT_URL, parquet)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = load_token_windows(tokenizer, parquet, args.windows,
                                args.sequence_length, args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    rows, summary = make_rows(model, tokenizer, inputs, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({"dataset": "WikiText-2 raw test split", "windows": len(inputs),
                    "sequence_length": args.sequence_length,
                    "wall_seconds": time.time() - start})
    with (out.parent / "run_summary.txt").open("w") as stream:
        for key, value in summary.items():
            stream.write(f"{key}: {value}\n")
    print(summary)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
