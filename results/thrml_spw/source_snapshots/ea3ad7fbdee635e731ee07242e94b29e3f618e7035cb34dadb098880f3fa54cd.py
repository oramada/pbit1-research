"""Label-free scalar moment comparison, at all three fixed equal-update budgets."""
import datetime
import json
from pathlib import Path
import numpy as np
from results.thrml_spw.cifar_cnn_native.cohort.common import ROOT, TRAIN, sha, write, save, guard
from results.thrml_spw.fashion_cnn_native.inference.native import encode as single_encode
from results.thrml_spw.fashion_cnn_native.word_reference.native import encode as word_encode

OUT = ROOT/'representation_moment_evidence'


def main():
    assert not OUT.exists()
    parent = json.loads((ROOT/'word_cohort_evidence/protocol.json').read_text())
    guard(parent)
    OUT.mkdir()
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_sha256': {**parent['source_sha256'], str(Path(__file__)): sha(__file__)},
                'input_sha256': parent['input_sha256'],
                'single_samples': [64, 256, 1024], 'word_samples': [16, 64, 256],
                'scope': 'Label-free nominal scalar weight means/bias/variance/MSE under independent temporal draws and F64sigmoid of actual programmedF32fields. Not measured task quality, actual F32sampler empirical moments, correlation/drift robustness, neuron error, sample-saving proof or hardware advantage. Every model/budget retained.'}
    write(OUT/'protocol.json', protocol)
    arrays, rows = {}, []
    train = json.loads((TRAIN/'manifest.json').read_text())['rows']
    for mi, row in enumerate(train):
        model = np.load(row['path'])
        singles, words = single_encode(model), word_encode(model)
        for li, prefix in enumerate(['conv1', 'conv2', 'head']):
            w = model[prefix+'_weight'].astype(float).reshape(len(model[prefix+'_weight']), -1).T
            s, word = singles[li], words[li]
            c, r = word['center'], word['radius']
            np.testing.assert_array_equal(s['center'], c)
            np.testing.assert_array_equal(s['radius'], r)
            ps = 1/(1+np.exp(-2*s['fields'].astype(float)))
            pw = 1/(1+np.exp(-2*word['fields'].astype(float)))
            ms = c-r+2*r*ps
            mw = c-r+2*r*np.sum(pw*np.array([1, 2, 4, 8]), axis=-1)/15
            vs = 4*r*r*ps*(1-ps)
            vw = 4*r*r*np.sum(pw*(1-pw)*np.array([1, 4, 16, 64]), axis=-1)/225
            key = f'model{mi}_layer{li}'
            for name, value in [('target', w), ('center', c), ('radius', r),
                                ('single_probabilities', ps), ('word_probabilities', pw),
                                ('single_mean', ms), ('word_mean', mw),
                                ('single_variance_per_round', vs), ('word_variance_per_round', vw)]:
                arrays[key+'_'+name] = value
        for ts, tw in zip(protocol['single_samples'], protocol['word_samples']):
            bs, bw, vs, vw = [], [], [], []
            for li in range(3):
                key = f'model{mi}_layer{li}_'
                w = arrays[key+'target']
                bs.extend(((arrays[key+'single_mean']-w)**2).ravel())
                bw.extend(((arrays[key+'word_mean']-w)**2).ravel())
                vs.extend((arrays[key+'single_variance_per_round']/ts).ravel())
                vw.extend((arrays[key+'word_variance_per_round']/tw).ravel())
            bs, bw, vs, vw = map(np.asarray, [bs, bw, vs, vw])
            assert len(bs) == 3928
            rows.append({'model': mi, 'T_single': ts, 'T_word': tw,
                         'single_mean_squared_bias': float(bs.mean()), 'word_mean_squared_bias': float(bw.mean()),
                         'single_mean_variance': float(vs.mean()), 'word_mean_variance': float(vw.mean()),
                         'single_mean_MSE': float((bs+vs).mean()), 'word_mean_MSE': float((bw+vw).mean()),
                         'fraction_weights_single_MSE_lower': float(np.mean(bs+vs < bw+vw))})
    save(OUT/'arrays.npz', **arrays)
    guard(protocol)
    write(OUT/'moments.json', {'rows': rows, 'arrays_sha256': sha(OUT/'arrays.npz'), 'scope': protocol['scope']})
    print('All19640weightconditions/15modelbudget summaries retained; independent moment verification pending', flush=True)


if __name__ == '__main__':
    main()
