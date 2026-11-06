"""Descriptive loss accounting for complete, independently accepted cases.

This does not evaluate the cohort gate, estimate uncertainty, or select a model.
The two differences are an algebraic decomposition, not causal attribution.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


STAGES = ("BF16", "same_fit_BF16", "nominal_controls", "native")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def components(means):
    total = means["native"] - means["same_fit_BF16"]
    stochastic = means["native"] - means["nominal_controls"]
    nominal = means["nominal_controls"] - means["same_fit_BF16"]
    assert total == stochastic + nominal
    return {
        "native_minus_same_fit_BF16_NLL": str(total),
        "native_minus_nominal_NLL": str(stochastic),
        "nominal_minus_same_fit_BF16_NLL": str(nominal),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/thrml_spw/lfm3_1p7b"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_sha = sha(Path(__file__))
    archived = Path("results/thrml_spw/source_snapshots") / (source_sha + ".py")
    assert sha(archived) == source_sha
    protocol_path = args.root / "counted_mean_validation_protocol.json"
    protocol = read(protocol_path)
    pipeline = read(args.root / "counted_mean_validation/pipeline.json")
    assert pipeline["protocol_sha256"] == sha(protocol_path)
    assert len(set(args.seeds)) == len(args.seeds)
    assert set(args.seeds) <= set(protocol["seeds"])
    verifier = Path(__file__).with_name("verify_counted_mean_validation.py")
    assert sha(verifier) == protocol["sources_sha256"][verifier.name]
    cases = []
    controls = None
    with localcontext() as context:
        context.prec = 80
        for seed in args.seeds:
            assert f"{seed}_verify_counted_mean_validation" in pipeline["completed"]
            directory = args.root / "counted_mean_validation/mean" / str(seed)
            audit_path = directory / "independent_audit.json"
            result_path = directory / "results.json"
            audit, result = read(audit_path), read(result_path)
            assert audit["status"] == "all_direct_program_coverage_losses_and_resource_receipts_verified"
            assert audit["source_sha256"] == sha(verifier)
            assert audit["results_sha256"] == sha(result_path)
            assert audit["seed"] == result["seed"] == seed
            assert audit["arm"] == result["arm"] == "mean"
            assert audit["targets"] == result["targets"] == 2048
            assert audit["unique_parameter_coordinates"] == 1728801536
            assert len(result["metrics"]) == 16
            losses = {}
            for stage in STAGES:
                rows = sorted((m for m in result["metrics"] if m["stage"] == stage), key=lambda m: m["batch"])
                assert [m["batch"] for m in rows] == list(range(4))
                assert all(len(m["token_NLL"]) == 8 for m in rows)
                assert all(len(row) == 64 for m in rows for row in m["token_NLL"])
                losses[stage] = [
                    [Decimal(str(v)) for v in row]
                    for m in rows for row in m["token_NLL"]
                ]
                assert all(v.is_finite() for row in losses[stage] for v in row)
            current_controls = {s: losses[s] for s in STAGES if s != "native"}
            if controls is None:
                controls = current_controls
            else:
                assert current_controls == controls
            means = {s: sum(sum(row) for row in losses[s]) / Decimal(2048) for s in STAGES}
            for s in STAGES:
                assert abs(means[s] - Decimal(str(audit["losses"][s]["mean_NLL_FP64"]))) < Decimal("1e-12")
            windows = []
            for index in range(32):
                window_means = {s: sum(losses[s][index]) / Decimal(64) for s in STAGES}
                windows.append({
                    "validation_window_ordinal": index,
                    "batch": index // 8,
                    "within_batch": index % 8,
                    "targets": 64,
                    "stage_NLL": {s: str(v) for s, v in window_means.items()},
                    **components(window_means),
                })
            for key, value in components(means).items():
                assert sum(Decimal(w[key]) for w in windows) / Decimal(32) == Decimal(value)
            cases.append({
                "seed": seed,
                "audit_sha256": sha(audit_path),
                "results_sha256": sha(result_path),
                "stage_NLL": {s: str(v) for s, v in means.items()},
                **components(means),
                "all_32_validation_windows": windows,
            })
    output = {
        "status": "accepted_complete_cases_descriptively_decomposed_no_cohort_inference",
        "source_sha256": source_sha,
        "protocol_sha256": sha(protocol_path),
        "accepted_seed_count": len(cases),
        "planned_seed_count": len(protocol["seeds"]),
        "cases": cases,
        "cohort_quality_gate_evaluated": False,
        "scope": (
            "Every saved token loss from complete accepted cases, all windows retained in original order. "
            "Native-minus-nominal includes stochastic trajectory effects and propagated activation-quantization effects; "
            "nominal-minus-same-fit-BF16 describes the deterministic programmed-weight and operator approximation. "
            "These are algebraic differences, not independent causal contributions or variance estimates. "
            "An expected-weight nonlinear network is not the expectation of a sampled network. "
            "No partial-cohort confidence interval, stopping rule, new selection, generalization or hardware claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
