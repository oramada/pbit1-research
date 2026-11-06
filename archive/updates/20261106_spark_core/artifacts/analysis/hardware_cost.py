"""Transparent first-order p-bit cost model.

All default device figures are illustrative knobs, not measurements. The model
separates transition energy, static bias programming, parallel-array latency,
readout, and digital range metadata. It intentionally makes no quality claim.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareAssumptions:
    transition_energy_pj: float = 1.0
    bias_program_energy_pj: float = 100.0
    sampling_frequency_hz: float = 10e6
    program_latency_s: float = 1e-6
    readout_latency_s: float = 1e-6
    reuse_inferences: int = 1000
    metadata_bytes_per_group: int = 8  # two FP32 scalars: center and radius

    def __post_init__(self) -> None:
        if self.transition_energy_pj < 0 or self.bias_program_energy_pj < 0:
            raise ValueError("energy assumptions must be nonnegative")
        if self.sampling_frequency_hz <= 0 or self.reuse_inferences < 1:
            raise ValueError("sampling frequency and reuse must be positive")


def evaluate_point(*, name: str, n_parameters: int, pbits_per_parameter: int,
                   samples: int, n_groups: int,
                   assumptions: HardwareAssumptions) -> dict[str, float | int | str]:
    """Calculate device, transition, energy, and latency metrics for one condition."""
    physical_pbits = int(n_parameters * pbits_per_parameter)
    metadata_bytes = int(n_groups * assumptions.metadata_bytes_per_group)
    transition_count = int(physical_pbits * samples)
    dynamic_energy = transition_count * assumptions.transition_energy_pj
    programming_energy = physical_pbits * assumptions.bias_program_energy_pj
    amortized_programming = programming_energy / assumptions.reuse_inferences
    sample_latency = samples / assumptions.sampling_frequency_hz
    first_latency = sample_latency + assumptions.program_latency_s + assumptions.readout_latency_s
    steady_latency = sample_latency + assumptions.readout_latency_s
    return {
        "condition": name,
        "parameters": n_parameters,
        "pbits_per_parameter": pbits_per_parameter,
        "samples": samples,
        "pbit_devices": physical_pbits,
        "metadata_groups": n_groups,
        "metadata_bytes": metadata_bytes,
        "metadata_bits_per_parameter": metadata_bytes * 8 / n_parameters if n_parameters else 0.0,
        "device_transitions_per_inference": transition_count,
        "dynamic_energy_pj_per_inference": dynamic_energy,
        "program_energy_pj_per_load": programming_energy,
        "amortized_program_energy_pj_per_inference": amortized_programming,
        "total_energy_pj_per_inference": dynamic_energy + amortized_programming,
        "first_inference_latency_us": first_latency * 1e6,
        "steady_latency_us": steady_latency * 1e6,
    }


def make_comparison(n_parameters: int = 1_000_000,
                    n_groups: int = 1_000,
                    assumptions: HardwareAssumptions | None = None) -> list[dict[str, float | int | str]]:
    """Compare hypothetical K=4 and SPW at equal samples and transitions.

    Equal sample count asks about spatial/device reduction at equal per-device
    temporal averaging. Equal transition count sets T_SPW = 4*T_K4.
    Neither comparison implies equal task quality.
    """
    hw = assumptions or HardwareAssumptions()
    rows = []
    for t_multi in (1, 4, 16, 64):
        t_single_equal_samples = t_multi
        rows.append(evaluate_point(name=f"equal_samples_K4_T{t_multi}",
                                   n_parameters=n_parameters, pbits_per_parameter=4,
                                   samples=t_multi, n_groups=0, assumptions=hw))
        rows.append(evaluate_point(name=f"equal_samples_SPW_T{t_single_equal_samples}",
                                   n_parameters=n_parameters, pbits_per_parameter=1,
                                   samples=t_single_equal_samples, n_groups=n_groups,
                                   assumptions=hw))
        t_single_equal_transitions = 4 * t_multi
        rows.append(evaluate_point(name=f"equal_transitions_K4_T{t_multi}",
                                   n_parameters=n_parameters, pbits_per_parameter=4,
                                   samples=t_multi, n_groups=0, assumptions=hw))
        rows.append(evaluate_point(name=f"equal_transitions_SPW_T{t_single_equal_transitions}",
                                   n_parameters=n_parameters, pbits_per_parameter=1,
                                   samples=t_single_equal_transitions, n_groups=n_groups,
                                   assumptions=hw))
    return rows
