"""Export K4/K8 comparison controls from the exact accepted adapted program.

CPU data preparation only; does not run a spatial neural evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .adapted_word_conversion import convert_codes
from .program_export import load_program


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("results/thrml_spw/lfm3_1p7b")
    origin = root / "adaptation_pilot/mean"
    package_path = origin / "program_package.json"
    package = json.loads(package_path.read_text())
    lower_protocol = json.loads((root / "counted_mean_validation_protocol.json").read_text())
    input_paths = [package_path, origin / "independent_audit.json"]
    for path in input_paths:
        assert sha(path) == lower_protocol["inputs_sha256"][str(path.relative_to(root))]
    aliases = origin / "module_aliases.json"
    assert sha(aliases) == package["module_aliases_sha256"]
    preflight_path = root / "adapted_word_conversion_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    assert preflight["status"] == "all_4096_field_codes_at_three_word_widths_match_independent_decimal_oracle"
    sources = {}
    for name in ["adapted_word_package.py", "adapted_word_conversion.py", "program_export.py"]:
        path = Path(__file__).with_name(name)
        sources[name] = sha(path)
        assert sha(Path("results/thrml_spw/source_snapshots") / (sources[name] + ".py")) == sources[name]
    assert preflight["sources_sha256"]["adapted_word_conversion.py"] == sources["adapted_word_conversion.py"]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"sources_sha256": sources, "original_package_sha256": sha(package_path),
                "original_training_audit_sha256": sha(input_paths[1]), "aliases_sha256": sha(aliases),
                "conversion_preflight_sha256": sha(preflight_path), "widths": [4, 8],
                "scope": "Exact adapted mean point, unchanged original lower/span, canonical shapes and tied aliases. Calibrated finite +/-4 positional-word comparison format. Actual uint8 word arrays are unpacked even for K4; compressed files do not measure hardware memory. No native sampling, model quality, physical residency, latency or energy result."}
    write_new(args.output / "protocol.json", protocol)
    with (args.output / "module_aliases.json").open("xb") as handle:
        handle.write(aliases.read_bytes())
    assert len(package["programs"]) == 286
    assert len({p["name"] for p in package["programs"]}) == 286
    lookup = {k: convert_codes(np.arange(4096, dtype=np.uint16), k) for k in [4, 8]}
    records = []
    total = metadata = 0
    for ordinal, record in enumerate(package["programs"]):
        path = Path(record["program"])
        assert sha(path) == record["program_sha256"]
        arrays = load_program(path)
        assert list(arrays["codes"].shape) == record["shape"]
        assert arrays["codes"].size == record["numel"]
        total += arrays["codes"].size
        metadata += arrays["lower"].nbytes + arrays["span"].nbytes
        outputs = {}
        for bits in [4, 8]:
            words = lookup[bits][arrays["codes"]]
            folder = args.output / f"K{bits}"
            folder.mkdir(exist_ok=True)
            target = folder / f"p{ordinal:03d}.npz"
            with target.open("xb") as handle:
                np.savez_compressed(handle, words=words, lower=arrays["lower"], span=arrays["span"],
                                    shape=arrays["shape"], group_size=arrays["group_size"],
                                    bits=np.asarray(bits, dtype=np.int64),
                                    format_version=np.asarray(2, dtype=np.int64),
                                    sign_field_bound=np.asarray(4, dtype=np.int64))
            outputs[str(bits)] = {"path": str(target), "sha256": sha(target),
                                  "words_sha256": hashlib.sha256(words.tobytes()).hexdigest(),
                                  "unpacked_word_bytes": words.nbytes, "serialized_bytes": target.stat().st_size}
        records.append({"name": record["name"], "shape": record["shape"], "numel": record["numel"],
                        "original_program": str(path), "original_program_sha256": record["program_sha256"],
                        "outputs": outputs})
        (args.output / "progress.json").write_text(json.dumps({"status": "exporting", "completed_programs": ordinal + 1,
                                                               "unique_coordinates_exported": total}) + "\n")
    assert total == 1728801536
    write_new(args.output / "package.json", {"status": "all_adapted_spatial_word_arrays_exported_pending_independent_audit",
              "protocol_sha256": sha(args.output / "protocol.json"), "unique_parameters": total,
              "canonical_programs": len(records), "shared_range_metadata_bytes_per_width": metadata,
              "actual_unpacked_word_bytes_per_width": total,
              "packed_sign_information_bits_per_width": {str(k): k * total for k in [4, 8]},
              "programs": records, "scope": protocol["scope"]})
    (args.output / "progress.json").write_text(json.dumps({"status": "export_complete_pending_independent_audit",
                                                           "completed_programs": len(records)}) + "\n")


if __name__ == "__main__":
    main()
