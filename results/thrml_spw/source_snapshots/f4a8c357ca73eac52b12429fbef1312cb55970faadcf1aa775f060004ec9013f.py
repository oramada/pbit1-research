"""Independently audit every adapted word and metadata byte using a Decimal LUT."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import numpy as np
from .adapted_word_package import sha, write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__)
    assert sha(Path("results/thrml_spw/source_snapshots") / (sha(source) + ".py")) == sha(source)
    protocol = json.loads((args.input / "protocol.json").read_text())
    package = json.loads((args.input / "package.json").read_text())
    assert package["protocol_sha256"] == sha(args.input / "protocol.json")
    for name, digest in protocol["sources_sha256"].items():
        assert sha(source.with_name(name)) == digest
    origin = Path("results/thrml_spw/lfm3_1p7b/adaptation_pilot/mean")
    assert sha(origin / "program_package.json") == protocol["original_package_sha256"]
    assert sha(origin / "independent_audit.json") == protocol["original_training_audit_sha256"]
    assert (args.input / "module_aliases.json").read_bytes() == (origin / "module_aliases.json").read_bytes()
    assert sha(args.input / "module_aliases.json") == protocol["aliases_sha256"]
    original_package = json.loads((origin / "program_package.json").read_text())
    assert len(package["programs"]) == len(original_package["programs"]) == 286
    with localcontext() as ctx:
        ctx.prec = 70
        one = Decimal(1)
        low, high = one / (one + Decimal(8).exp()), one / (one + Decimal(-8).exp())
        fields = np.add(np.float32(-4), np.multiply(np.arange(4096, dtype=np.float32), np.float32(8 / 4095)))
        probabilities = [one / (one + (-Decimal(2) * Decimal.from_float(float(v))).exp()) for v in fields]
        tables = {}
        for bits in [4, 8]:
            lattice = [low + (high - low) * Decimal(i) / Decimal((1 << bits) - 1) for i in range(1 << bits)]
            tables[bits] = np.array([min(range(len(lattice)), key=lambda i: abs(lattice[i] - p)) for p in probabilities], dtype=np.uint8)
    total = metadata = 0
    for record, original in zip(package["programs"], original_package["programs"]):
        assert record["name"] == original["name"] and record["shape"] == original["shape"]
        assert record["numel"] == original["numel"]
        path = Path(record["original_program"])
        assert str(path) == original["program"]
        assert sha(path) == original["program_sha256"] == record["original_program_sha256"]
        with np.load(path) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
        codes = arrays["codes"]
        assert codes.dtype == np.uint16 and np.max(codes) <= 4095
        assert codes.size == record["numel"] and list(codes.shape) == record["shape"]
        total += codes.size
        metadata += arrays["lower"].nbytes + arrays["span"].nbytes
        for bits in [4, 8]:
            item = record["outputs"][str(bits)]
            exported = Path(item["path"])
            assert exported.resolve().parent == (args.input / f"K{bits}").resolve()
            assert sha(exported) == item["sha256"] and exported.stat().st_size == item["serialized_bytes"]
            with np.load(exported) as word:
                assert set(word.files) == {"words", "lower", "span", "shape", "group_size", "bits", "format_version", "sign_field_bound"}
                values = word["words"]
                assert values.dtype == np.uint8 and values.shape == codes.shape
                assert int(word["bits"]) == bits and int(word["format_version"]) == 2 and int(word["sign_field_bound"]) == 4
                for name in ["lower", "span", "shape", "group_size"]:
                    a, b = word[name], arrays[name]
                    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                for start in range(0, codes.size, 1048576):
                    expected = tables[bits][codes.reshape(-1)[start:start + 1048576]]
                    np.testing.assert_array_equal(values.reshape(-1)[start:start + 1048576], expected)
                assert values.nbytes == item["unpacked_word_bytes"] == codes.size
                assert hashlib.sha256(values.tobytes()).hexdigest() == item["words_sha256"]
    assert total == package["unique_parameters"] == 1728801536
    assert metadata == package["shared_range_metadata_bytes_per_width"]
    assert package["actual_unpacked_word_bytes_per_width"] == total
    assert package["packed_sign_information_bits_per_width"] == {str(k): k * total for k in [4, 8]}
    write_new(args.input / "independent_audit.json", {"status": "every_K4_K8_word_and_original_metadata_byte_verified",
              "source_sha256": sha(source), "package_sha256": sha(args.input / "package.json"),
              "unique_parameters_per_width": total, "word_coordinates_checked": 2 * total,
              "canonical_programs_per_width": 286, "shared_range_metadata_bytes_per_width": metadata,
              "actual_unpacked_word_bytes_per_width": total,
              "scope": "All canonical adapted word controls and tied alias bytes checked against the accepted original program, using an independently enumerated 70-digit Decimal nearest-lattice oracle. No native sampling, task quality or hardware-performance validation."})


if __name__ == "__main__":
    main()
