from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analysis.hardware_cost import HardwareAssumptions, make_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=int, default=1_000_000)
    parser.add_argument("--groups", type=int, default=1_000)
    parser.add_argument("--transition-energy-pj", type=float, default=1.0)
    parser.add_argument("--program-energy-pj", type=float, default=100.0)
    parser.add_argument("--sampling-hz", type=float, default=10e6)
    parser.add_argument("--program-latency-us", type=float, default=1.0)
    parser.add_argument("--readout-latency-us", type=float, default=1.0)
    parser.add_argument("--reuse", type=int, default=1000)
    parser.add_argument("--out", default="results/cost_model.csv")
    args = parser.parse_args()
    assumptions = HardwareAssumptions(
        transition_energy_pj=args.transition_energy_pj,
        bias_program_energy_pj=args.program_energy_pj,
        sampling_frequency_hz=args.sampling_hz,
        program_latency_s=args.program_latency_us * 1e-6,
        readout_latency_s=args.readout_latency_us * 1e-6,
        reuse_inferences=args.reuse,
    )
    rows = make_comparison(args.parameters, args.groups, assumptions)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    print("ASSUMPTIONS (illustrative, not device measurements):")
    print(f"  transition={assumptions.transition_energy_pj:g} pJ/device-sample; "
          f"bias program={assumptions.bias_program_energy_pj:g} pJ/device/load")
    print(f"  sample rate={assumptions.sampling_frequency_hz:g} Hz; "
          f"program={assumptions.program_latency_s*1e6:g} us; "
          f"readout={assumptions.readout_latency_s*1e6:g} us; "
          f"weight reuse={assumptions.reuse_inferences} inferences")
    for row in rows:
        print(f"{row['condition']:33s} devices={row['pbit_devices']:>9,d} "
              f"transitions={row['device_transitions_per_inference']:>12,d} "
              f"E={row['total_energy_pj_per_inference']/1e6:>8.3f} uJ "
              f"Lsteady={row['steady_latency_us']:>8.3f} us")


if __name__ == "__main__":
    main()
