"""Exhaustive CPU oracle for finite-bias adapted positional-word conversion."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np
from .adapted_word_conversion import convert_codes, expected_probability, expected_word_probability


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__)
    conversion = source.with_name("adapted_word_conversion.py")
    for path in (source, conversion):
        archived = Path("results/thrml_spw/source_snapshots") / (digest(path) + ".py")
        assert digest(archived) == digest(path)
    codes = np.arange(4096, dtype=np.uint16)
    # Separate target-law arithmetic, matching the documented float32 field map.
    fields = np.add(np.float32(-4), np.multiply(codes.astype(np.float32), np.float32(8 / 4095)))
    records = []
    with localcontext() as ctx:
        ctx.prec = 70
        one = Decimal(1)
        p0 = one / (one + Decimal(8).exp())
        p1 = one / (one + Decimal(-8).exp())
        targets = [one / (one + (-Decimal(2) * Decimal.from_float(float(f))).exp()) for f in fields]
        np.testing.assert_allclose(expected_probability(codes), np.array([float(v) for v in targets]), rtol=0, atol=3e-16)
        for bits in (2, 4, 8):
            words = convert_codes(codes, bits)
            levels = (1 << bits) - 1
            lattice = [p0 + (p1 - p0) * Decimal(i) / Decimal(levels) for i in range(levels + 1)]
            oracle = np.array([min(range(levels + 1), key=lambda i: abs(lattice[i] - target)) for target in targets], dtype=np.uint8)
            np.testing.assert_array_equal(words, oracle)
            actual = expected_word_probability(words, bits)
            wanted = np.array([float(lattice[int(w)]) for w in words])
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=3e-16)
            assert words[0] == 0 and words[-1] == levels
            assert np.all(np.diff(words.astype(np.int16)) >= 0)
            max_error = max(abs(lattice[int(w)] - t) for w, t in zip(words, targets))
            assert max_error <= (p1 - p0) / (2 * Decimal(levels))
            records.append({"bits": bits, "all_4096_command_words_match_decimal_nearest_lattice": True,
                            "maximum_probability_error_decimal": str(max_error),
                            "half_lattice_step_decimal": str((p1 - p0) / (2 * Decimal(levels))),
                            "word_sha256": hashlib.sha256(words.tobytes()).hexdigest()})
    invalid = 0
    for codes_arg, bits in [(codes, 1), (codes, 3), (codes.astype(np.int32), 4), (np.array([4096], np.uint16), 4), (np.array([], np.uint16), 8)]:
        try:
            convert_codes(codes_arg, bits)
        except ValueError:
            invalid += 1
        else:
            raise AssertionError("Invalid input accepted")
    output = {"status": "all_4096_field_codes_at_three_word_widths_match_independent_decimal_oracle",
              "sources_sha256": {source.name: digest(source), conversion.name: digest(conversion)},
              "cases": records, "invalid_inputs_rejected": invalid,
              "scope": "CPU encoding-law check only. Original lower/span metadata unchanged by this array converter. Finite +/-4 sign controls are calibrated to approximate the same original expected probability, avoiding a second endpoint contraction. No adapted tensor package export, sampler trajectory, full-model quality, matched-work or physical hardware result. New native preflight and a separately frozen adapted spatial study remain required."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
