"""Audit raw stronger quantizer scores and their explicit packed metadata."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np


def main():
    out=Path("results/thrml_spw/block_quantization_controls")
    raw=np.load(out/"tokens.npz")
    data=np.load("data/thrml_spw/transformer_projection/export.npz")
    with (out/"metrics.csv").open() as f:rows=list(csv.DictReader(f))
    for row in rows:
        key=row["split"]+"_"+row["condition"]
        nll,pred=raw[key+"_nll"],raw[key+"_predictions"]
        ce=float(nll.astype(np.float64).mean())
        np.testing.assert_allclose(ce,float(row["cross_entropy"]),atol=1e-12)
        np.testing.assert_allclose(np.exp(ce),float(row["perplexity"]),rtol=1e-12)
        np.testing.assert_allclose(np.mean(pred==data[row["split"]+"_tokens"][:,1:]),float(row["token_accuracy"]),atol=1e-12)
        bits,groups=int(row["bits"]),int(row["groups"])
        assert int(row["nominal_code_bits"])==7077888*bits
        assert groups==(55296 if int(row["input_block_size"]) else 6912)
        assert int(row["nominal_scale_bits"])==32*groups
        assert int(row["nominal_zero_point_bits"])==(bits*groups if row["method"]=="affine_zero" else 0)
    result={"passed":True,"conditions":len(rows),"scope":"Raw token losses/accuracy and nominal code/scale/zero-point counts for all weights-only controls; no optimized-kernel or corpus-wide claim"}
    (out/"evidence_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
