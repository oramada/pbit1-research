"""Full block quality, honest low-bit controls, costs and localization forecasts."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

ROOT=Path("results/thrml_spw")


def rows(path):
    with (ROOT/path).open() as f:return list(csv.DictReader(f))


def main():
    block=ROOT/"transformer_block0_gpu"
    manifest=json.loads((block/"manifest.json").read_text())
    protocol=manifest["config"];selection=manifest["selection"]
    summary=rows("transformer_block0_gpu/summary.csv")
    baselines=rows("transformer_block0_gpu/baseline_metrics.csv")
    controls=rows("block_quantization_controls/metrics.csv")
    prefixes=rows("transformer_block0_gpu/prefix_metrics.csv")
    forecasts=rows("block_range_analysis/metrics.csv")
    fp={r["split"]:r for r in baselines if r["condition"]=="FP32"}
    selected=next(r for r in summary if r["split"]=="test" and int(r["samples"])==selection["selected"]["samples"])
    records=[]
    sample_run=json.loads((block/f"runs/test_T{selection['selected']['samples']}_seed0/run.json").read_text())
    metadata=[json.loads(Path(s["metadata"]).read_text()) for s in sample_run["stages"]]
    n=protocol["converted_matrix_weights"];biases=protocol["deterministic_converted_module_biases"]
    field_bytes=sum(m["controls"]["bias_storage_bytes_simulator"] for m in metadata)
    range_bits=sum(m["controls"]["metadata_bits"] for m in metadata)
    native_subtotal=field_bytes*8+range_bits+biases*32
    records.append({"condition":"FP32","logical_pbit_nodes":0,"numeric_or_field_bits":n*32,"scale_range_bits":0,"zero_point_bits":0,"bias_bits":biases*32,
                    "declared_subtotal_bits":n*32+biases*32,"scope":"Nominal numeric values, not measured Python memory or area"})
    records.append({"condition":"native_float32_controls","logical_pbit_nodes":n,"numeric_or_field_bits":field_bytes*8,"scale_range_bits":range_bits,"zero_point_bits":0,"bias_bits":biases*32,
                    "declared_subtotal_bits":native_subtotal,"scope":"Field/range/bias subtotal; excludes p-bit states, endpoint modes, requested probability arrays, program objects, accumulators and all other buffers"})
    for r in controls:
        if r["split"]!="test":continue
        code,scale,zero=[int(r[k]) for k in ("nominal_code_bits","nominal_scale_bits","nominal_zero_point_bits")]
        records.append({"condition":r["condition"],"logical_pbit_nodes":0,"numeric_or_field_bits":code,"scale_range_bits":scale,"zero_point_bits":zero,"bias_bits":biases*32,
                        "declared_subtotal_bits":code+scale+zero+biases*32,"scope":"Nominal packed code/scale/integer-zero-point storage; numerical dequantized inference, not packed-kernel execution"})
    with (block/"storage_ledger.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    for split,color in (("validation","tab:blue"),("test","tab:orange")):
        group=sorted([r for r in summary if r["split"]==split],key=lambda r:int(r["samples"]))
        x=[int(r["samples"]) for r in group];y=[float(r["relative_perplexity"]) for r in group]
        lower=[v-float(r["relative_perplexity_lower"]) for r,v in zip(group,y)]
        upper=[float(r["relative_perplexity_upper"])-v for r,v in zip(group,y)]
        axes[0,0].errorbar(x,y,yerr=[lower,upper],fmt="o-",color=color,label=split)
    axes[0,0].axhline(1.05,color="black",ls="--",label="5% calibration tolerance")
    axes[0,0].set(xscale="log",yscale="log",xlabel="Samples per projection",ylabel="Perplexity / FP32",title="All four projection matrices converted")
    axes[0,0].set_xticks([64,256,1024]);axes[0,0].set_xticklabels([64,256,1024]);axes[0,0].xaxis.set_minor_formatter(NullFormatter());axes[0,0].legend(fontsize=8)
    for t in sorted({int(r["samples"]) for r in prefixes if r["split"]=="validation"}):
        curve=[]
        for prefix in range(5):
            g=[r for r in prefixes if r["split"]=="validation" and int(r["samples"])==t and int(r["converted_prefix"])==prefix]
            curve.append(np.exp(np.mean([float(r["cross_entropy"]) for r in g])-float(fp["validation"]["cross_entropy"])))
        axes[0,1].plot(range(5),curve,"o-",label=f"T={t}")
    axes[0,1].set(xticks=range(5),xticklabels=["FP32","+QKV","+attn out","+MLP up","+MLP down"],ylabel="Calibration perplexity / FP32",title="Incremental effects along this conversion order")
    axes[0,1].legend(fontsize=8)
    labels=["FP32","native1024","sym output4","zero output4","sym block128","zero block128"]
    values=[1,float(selected["relative_perplexity"])]
    for name in ("symmetric_output4","affine_zero_output4","symmetric_block128_4","affine_zero_block128_4"):
        r=next(r for r in controls if r["split"]=="test" and r["condition"]==name)
        values.append(np.exp(float(r["cross_entropy"])-float(fp["test"]["cross_entropy"])))
    axes[1,0].bar(range(6),values,color=["black","tab:orange","tab:blue","tab:blue","tab:green","tab:green"])
    axes[1,0].set(xticks=range(6),xticklabels=labels,ylabel="Test perplexity / FP32",title="Zero-preserving low-bit controls also work")
    axes[1,0].tick_params(axis="x",labelrotation=35)
    axes[1,0].set_ylim(.98,max(values)+.02)
    ratios=[float(next(r for r in forecasts if r["inputs"]=="teacher" and int(r["stage"])==i and r["grouping"]=="input_block128")["variance_fraction_of_per_output"]) for i in range(4)]
    axes[1,1].bar(range(4),ratios,color="tab:purple")
    axes[1,1].set(xticks=range(4),xticklabels=["QKV","Attn out","MLP up","MLP down"],ylim=(0,1),ylabel="Predicted variance / per-output range",title="Analytical forecast only: input groups of 128")
    for ax in axes.flat:ax.grid(axis="y",alpha=.2)
    fig.suptitle("One DistilGPT-2 block: 7.08M native matrix weights (8.64% of model), five sampling seeds")
    fig.savefig(ROOT/"transformer_block_evidence.png",dpi=180);fig.savefig(ROOT/"transformer_block_evidence.pdf");plt.close(fig)
    lines=["# Native p-bit conversion of all projection matrices in one Transformer block\n",
        "All four projection matrices in DistilGPT-2 block0 run through official THRML: QKV, "
        "attention output, MLP up and MLP down. There are 7,077,888 converted matrix weights, "
        "8.6408% of the model's 81,912,576 unique parameters. Biases, LayerNorm, embeddings, "
        "the output head and the other five blocks remain deterministic. This is not a "
        "full-model conversion or physical probabilistic-hardware experiment.\n",
        "## Correct sequential execution\n",
        "Every probability bank has one logical node per normalized weight and per-output "
        "float32 lower/span metadata. Each native layer accumulates linear outputs before "
        "its downstream nonlinearity. Later projection inputs are regenerated by rerunning "
        "the real model after every earlier replacement. They are not fixed teacher inputs.\n",
        "The FP32 four-projection replay exactly recovers original token losses. Every native "
        "prefix checks its cached inputs exactly; an independent checker uses pre/post hooks "
        "rather than method replacement to recover every completed run's token losses. Raw "
        "student inputs, conditional FP32 references, native outputs and prefix token evidence "
        "are preserved with checksums and source/key signatures. Atomic stage checkpoints "
        "support safe resume and reject changed inputs.\n",
        "## Calibration and held-out results\n",
        "The ascending predeclared calibration search was 64, 256, 1024 samples per projection, "
        "five sampling seeds each. A gate requires at most 5% relative perplexity increase "
        "and at most one percentage-point token-accuracy loss. Calibration used 1,024 targets "
        "and the held-out test subset used 2,048, in separate official WikiText-2 splits and "
        "nonoverlapping short-context windows. These are subset scores, not full-corpus benchmarks.\n",
        "| Split | Samples per projection | exp(mean CE) perplexity | Relative to FP32, approximate 95% interval |\n|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['split']} | {r['samples']} | {float(r['exp_mean_cross_entropy']):.3f} | {float(r['relative_perplexity']):.4f} [{float(r['relative_perplexity_lower']):.4f},{float(r['relative_perplexity_upper']):.4f}] |")
    lines += [f"\nCalibration selected T={selection['selected']['samples']}. On test the mean "
        f"perplexity increase is {100*(float(selected['relative_perplexity'])-1):.2f}%. "
        "Low budgets failed: T64 calibration perplexity was about three times FP32, and "
        "T256 still increased it by 16.8%. Those failures were retained. Intervals apply to "
        "five Monte Carlo seeds of one fixed model/corpus subset, not population or training uncertainty.\n",
        "Incremental prefix curves diagnose this forward-order experiment. MLP projections "
        "add most of the degradation here; this is conditional on earlier replacements, not "
        "a universal ranking of isolated layer sensitivities.\n",
        "## Stronger conventional controls\n",
        "The original per-output affine4 lattice did not preserve zero and scored test "
        "perplexity 1188.55. That is not evidence that INT4 generally fails. Six additional "
        "weights-only controls were predeclared and all reported: symmetric/zero-point affine "
        "per-output and input-block128 formats. There is no label-based quantizer fitting.\n",
        "| Control | Test perplexity | Relative to FP32 |\n|---|---:|---:|"]
    for r in controls:
        if r["split"]=="test":lines.append(f"| {r['condition']} | {float(r['perplexity']):.3f} | {np.exp(float(r['cross_entropy'])-float(fp['test']['cross_entropy'])):.4f} |")
    lines += ["\nZero-preserving block128 affine4 reaches 112.806 versus native T1024 "
        "114.987 and FP32 112.207. Thus useful native quality is established, but no quality "
        "or systems superiority over conventional low-bit inference is established. These "
        "are basic numerical controls, not exhaustive GPTQ/AWQ methods or optimized packed kernels.\n",
        "## Resources\n",
        f"The selected test run executes {int(selected['free_node_updates']):,} free-node "
        f"evaluations per seed across four eight-context bank batches and "
        f"{int(selected['sequential_sampling_rounds']):,} dependent sampling rounds per complete block. "
        "Counts include only the declared operator schedule, not extra reference/audit computations. "
        "They are not measured p-bit transitions, chip latency or energy.\n",
        f"Per-output range metadata is {range_bits//8:,} bytes; retained selected-module "
        f"neural biases use {biases*4:,} bytes. Float32 control fields occupy a declared "
        f"{field_bytes:,}-byte subtotal. Fields plus range metadata and biases total "
        f"{native_subtotal//8:,} bytes, equal in this run to those matrices/biases in FP32, "
        "before adding p-bit states, endpoint modes, program/probability arrays, accumulators "
        "and all other buffers. Software compression is not achieved. storage_ledger.csv "
        "separates these counts from nominal packed digital codes/scales/zero points.\n",
        "CPU runs were kept as partial backend-reference evidence. A CUDA environment using "
        "the same official sampler, highest matmul precision and unchanged logical counts "
        "accelerated the full experiment. A large-operator backend comparison and all five "
        "T64 block calibration pairs agree to small numerical differences. The redundant CPU "
        "runner was then stopped, preserving completed checkpoints and an explicit partial "
        "status. GPU wall time is not a physical TSU performance claim.\n",
        "## Next hypotheses, not demonstrated quality gains\n",
        "A separate analytical sweep predicts conditional noise with per-matrix/per-output "
        "and smaller input groups. Input-block128 ranges predict roughly 0.23 of the current "
        "MLP-down variance at fixed saved teacher inputs, with 0.5 total metadata bits per "
        "weight versus 0.0625 currently (an increase of 0.4375 bits per weight). This forecast motivates a native "
        "grouping experiment; it is not a neural-quality result. Finite controller precision, "
        "calibration/drift/correlation, stronger training, larger converted fractions and "
        "physical controller/readout/accumulator costs remain open. The long-horizon goal stays active.\n"]
    (ROOT/"TRANSFORMER_BLOCK_REPORT.md").write_text("\n".join(lines)+"\n")
    print("Wrote TRANSFORMER_BLOCK_REPORT.md, storage_ledger.csv and transformer_block_evidence.png/.pdf")


if __name__=="__main__":main()
