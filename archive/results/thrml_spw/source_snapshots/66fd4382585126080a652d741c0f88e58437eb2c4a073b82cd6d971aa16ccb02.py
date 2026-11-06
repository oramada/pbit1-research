"""Shared file conventions for the closed-cohort fine-tuning figures."""
from pathlib import Path
import hashlib
import json

ROOT = Path('results/thrml_spw/fashion_cnn_native')
BASE = ROOT/'cohort_evidence'
WORD = ROOT/'word_comparison_evidence'
FINE = ROOT/'probability_cohort_evidence'
OUT = ROOT/'finetune_figure_evidence'
LABELS = ['Frozen: one p-bit', 'Expected-weight training', 'Sample-aware training', 'Frozen: four p-bits']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def guard(p):
    for mapping in [p['source_sha256'], p['input_sha256']]:
        for path, h in mapping.items():
            assert sha(path) == h, path


def seal(path):
    for file, h in json.loads(path.read_text())['artifact_sha256'].items():
        assert sha(file) == h, file
