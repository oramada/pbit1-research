"""Independent exhaustive two/16-state moment laws for fixed RGB parameters."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path('results/thrml_spw/cifar_cnn_native')
OUT = ROOT/'representation_moment_evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (OUT/'completion.json').exists()
    p = json.loads((OUT/'protocol.json').read_text())
    for kind in ['source_sha256', 'input_sha256']:
        for path, h in p[kind].items():
            assert sha(path) == h, path
    d = json.loads((OUT/'moments.json').read_text())
    assert sha(OUT/'arrays.npz') == d['arrays_sha256']
    arrays = np.load(OUT/'arrays.npz')
    train = json.loads((ROOT/'training/manifest.json').read_text())['rows']
    independent = {}
    count = 0
    for mi, row in enumerate(train):
        model = np.load(row['path'])
        for li, prefix in enumerate(['conv1', 'conv2', 'head']):
            weights = model[prefix+'_weight'].astype(float)
            targets = weights.reshape(len(weights), -1).T
            key = f'model{mi}_layer{li}_'
            np.testing.assert_array_equal(arrays[key+'target'], targets)
            sm, wm, sv, wv = [np.empty_like(targets) for _ in range(4)]
            for o in range(len(weights)):
                lo, hi = min(targets[:, o]), max(targets[:, o])
                c, r = (lo+hi)/2, (hi-lo)/2
                assert c == arrays[key+'center'][o] and r == arrays[key+'radius'][o]
                for i, target in enumerate(targets[:, o]):
                    q = (float(target)-c)/(2*r)+.5 if r else .5
                    eps = 1/(1+math.exp(8))
                    bounded = min(1-eps, max(eps, q))
                    command = round((math.log(bounded/(1-bounded))/2+4)*255/8)
                    field = float(np.float32(-4+command*8/255))
                    ps = 1/(1+math.exp(-2*field))
                    wordcode = round(min(1, max(0, q))*15)
                    probs = [1/(1+math.exp(-2*(4 if (wordcode >> bit)&1 else -4))) for bit in range(4)]
                    np.testing.assert_allclose(ps, arrays[key+'single_probabilities'][i, o], rtol=2e-14, atol=2e-15)
                    np.testing.assert_allclose(probs, arrays[key+'word_probabilities'][i, o], rtol=2e-14, atol=2e-15)
                    svalues, sprob = [c-r, c+r], [1-ps, ps]
                    states = [c-r+2*r*state/15 for state in range(16)]
                    masses = [math.prod(probs[bit] if (state >> bit)&1 else 1-probs[bit] for bit in range(4)) for state in range(16)]
                    assert abs(math.fsum(masses)-1) < 2e-15
                    sm[i, o] = math.fsum(a*b for a, b in zip(svalues, sprob))
                    wm[i, o] = math.fsum(a*b for a, b in zip(states, masses))
                    sv[i, o] = math.fsum(b*(a-sm[i, o])**2 for a, b in zip(svalues, sprob))
                    wv[i, o] = math.fsum(b*(a-wm[i, o])**2 for a, b in zip(states, masses))
                    count += 1
            for name, value in [('single_mean', sm), ('word_mean', wm), ('single_variance_per_round', sv), ('word_variance_per_round', wv)]:
                np.testing.assert_allclose(value, arrays[key+name], rtol=3e-11, atol=3e-15)
                independent[key+name] = value
    assert count == 19640 and len(d['rows']) == 15
    seen = set()
    for row in d['rows']:
        mi, ts, tw = row['model'], row['T_single'], row['T_word']
        assert (mi, ts) not in seen and ts == 4*tw
        seen.add((mi, ts))
        bm, bw, vs, vw = [], [], [], []
        for li in range(3):
            key = f'model{mi}_layer{li}_'
            w = arrays[key+'target']
            bm.extend(((independent[key+'single_mean']-w)**2).ravel())
            bw.extend(((independent[key+'word_mean']-w)**2).ravel())
            vs.extend((independent[key+'single_variance_per_round']/ts).ravel())
            vw.extend((independent[key+'word_variance_per_round']/tw).ravel())
        bm, bw, vs, vw = map(np.asarray, [bm, bw, vs, vw])
        for name, value in [('single_mean_squared_bias', bm.mean()), ('word_mean_squared_bias', bw.mean()),
                            ('single_mean_variance', vs.mean()), ('word_mean_variance', vw.mean()),
                            ('single_mean_MSE', (bm+vs).mean()), ('word_mean_MSE', (bw+vw).mean()),
                            ('fraction_weights_single_MSE_lower', np.mean(bm+vs < bw+vw))]:
            np.testing.assert_allclose(row[name], value, rtol=3e-11, atol=3e-15)
    assert seen == {(mi, ts) for mi in range(5) for ts in [64, 256, 1024]}
    report = ROOT.parent/'CIFAR_CNN_REPRESENTATION_MOMENTS_REPORT.md'
    assert not report.exists()
    lines = ['# Nominal scalar encoding error on the five RGB models', '',
        'Independent enumeration of two single-node outcomes and all16word outcomes checks every19640weight conditions and all15model/budget summaries. No labels, native draws or task metrics are used. Probabilities use the ideal F64sigmoid of programmedF32fields, not an empirical F32sampler/device characterization. Independent temporal draws and unchanged group centers/radii are assumed.', '',
        '| Single T | Word T | Single mean scalar MSE | Word mean scalar MSE | Fraction of weights with lower single MSE |',
        '|---:|---:|---:|---:|---:|']
    for ts, tw in [(64, 16), (256, 64), (1024, 256)]:
        rows = [r for r in d['rows'] if r['T_single'] == ts]
        lines.append(f"|{ts}|{tw}|{np.mean([r['single_mean_MSE'] for r in rows]):.10g}|{np.mean([r['word_mean_MSE'] for r in rows]):.10g}|{np.mean([r['fraction_weights_single_MSE_lower'] for r in rows]):.6f}|")
    lines += ['', 'The one-node variable probability and the four-node nearly saturated positional word trade sampling variance against finite-control/word quantization bias. Scalar MSE is squared bias plus per-round variance divided by the respective sample budget. Equal4*Tword=Tsingle matches retained node evaluations. This calculation does not establish neuron/task accuracy, statistical equivalence, a sampling benefit from linearity, physical controller cost, energy, latency or a systems win. Every predetermined budget/model is retained; no success gate is selected from these values.', '',
        '[All per-weight arrays](cifar_cnn_native/representation_moment_evidence/arrays.npz) · [All model/budget summaries](cifar_cnn_native/representation_moment_evidence/moments.json) · [Protocol and scope](cifar_cnn_native/representation_moment_evidence/protocol.json). Full native RGB quality and matched-word comparison remain separately audited tasks.']
    report.write_text('\n'.join(lines)+'\n')
    completion = {'passed': True, 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'checker_source_sha256': sha(__file__), 'weight_conditions_checked': count,
                  'word_states_per_condition': 16, 'model_budget_summaries_checked': 15,
                  'native_draws': 0, 'task_metrics': 0,
                  'artifact_sha256': {str(path): sha(path) for path in [OUT/'protocol.json', OUT/'moments.json', OUT/'arrays.npz', report]}}
    (OUT/'completion.json').write_text(json.dumps(completion, indent=2)+'\n')
    print('All19640nominalscalar conditions checked by separate two/16outcome enumeration', flush=True)


if __name__ == '__main__':
    main()
