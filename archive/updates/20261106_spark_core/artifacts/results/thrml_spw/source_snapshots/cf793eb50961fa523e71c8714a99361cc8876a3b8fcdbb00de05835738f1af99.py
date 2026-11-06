"""Executable distinctions between task quality and universal BF16 equality."""
import json,math
import numpy as np
from scipy.stats import binom
from .prepare_v1 import OUT,sha
from .weight_survey_v1 import mapped

def main():
    dest=OUT/'precision_limits_v1';dest.mkdir(exist_ok=False)
    a=np.full(128,.25,np.float32);a[0]=0.;a[1]=1.;a[2]=2.**-20
    b=a.copy();b[2]=2.**-19
    assert np.all((a.view(np.uint32)&65535)==0)and np.all((b.view(np.uint32)&65535)==0)
    pa=mapped(a,128);pb=mapped(b,128)
    for x,y in zip(pa,pb):np.testing.assert_array_equal(x,y)
    assert a[2]!=b[2]
    # .75 is an even-significand BF16 value with symmetric 1/256 neighbor spacing.
    # Both exactly-halfway endpoints round to .75 (round to nearest, ties to even).
    p=.75;half_spacing=1/512;N=124439808;rows=[]
    for T in [8192,32768,131072,524288,1048576]:
        lo=math.ceil(T*(p-half_spacing));hi=math.floor(T*(p+half_spacing))
        failure=float(binom.cdf(lo-1,T,p)+binom.sf(hi,T,p));success=1-failure
        rows.append({'samples':T,'single_weight_rounds_to_BF16_target_probability':success,
            'log10_probability_all_124439808_identical_independent_weights_round_correctly':N*math.log1p(-failure)/math.log(10),
            'expected_incorrectly_rounded_weights':N*failure})
    result={'source_sha256':sha(__file__),'mapping_sha256':sha(__import__('pathlib').Path(__file__).with_name('weight_survey_v1.py')),
        'finite_controller_counterexample':{'group_size':128,'range':[0,1],'target_A':float(a[2]),'target_B':float(b[2]),
            'all_target_values_exact_BF16':True,'entire_encoded_programs_identical':True,'code_at_changed_weight':int(pa[0][2]),'asymptotic_reconstruction_at_changed_weight':float(pa[3][2]),
            'meaning':'Current12bit[-4,4]finitefieldclippingcannotdistinguishtheseBF16weights inthisfixedrangegroup, evenwithunlimitedsamples. Differentgrouping/controllerprotocolcanchange thiscounterexample.'},
        'ideal_bernoulli_rounding_example':{'target':p,'BF16_neighbor_spacing':1/256,'controller':'Ideal exact p=.75, before finite-code error','results':rows,
            'meaning':'Illustrativetensorofindependentidenticalweights; notaneuralmodelandnotanapplicationqualityprediction.'},
        'finite_observation_argument':'For two different interior Bernoulliprobabilities, every finitebinarysequence haspositiveprobabilityunderboth. A decoder givenonlythoseobservationsandsharedrangecannotguarantee exactrecoveryofboth distincttargets. Readinganexactweightcopyfromcontrollerbypassesestimation andmustbecounted asstoredinformation.',
        'scope':'These disprove universal numerical equality underthespecifiedencodings, notthepossibilityofBF16-leveltaskquality oroneprogrammablepbitperscalar.'}
    (dest/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (dest/'README.md').write_text('''# Numerical precision and task quality are different requirements

A single programmable p-bit can encode a scalar in its stationary probability. This does not guarantee that a finite sample reproduces the scalar's BF16 bit pattern, or that a finite controller can program every BF16 value.

The executable counterexample holds the group range fixed at [0, 1]. Two arrays of exact BF16 values differ only at one coordinate (2^-20 versus 2^-19). The current 12-bit, field-clipped encoder produces identical complete codes and range metadata for both arrays. Its asymptotic output is therefore identical for two different targets. Changing the range/controller protocol may repair this particular collision, at a cost that must be counted.

Separately, even an ideal probability controller has finite-sample uncertainty. The accompanying exact binomial calculation gives the chance of rounding an estimate of p = 0.75 to the correct BF16 value. It is an illustrative numerical test, not a prediction of neural-network accuracy.

For distinct interior probabilities, the same finite binary observation sequence can occur under either one. Observations alone cannot guarantee exact recovery for both. If a decoder instead reads an exact stored parameter from the controller, that stored information must be counted.

None of these arguments rule out matching BF16 **task quality**, which remains the empirical research target. They delimit a stronger, universal numerical-equality claim.
''')
    print(json.dumps({'counterexample':result['finite_controller_counterexample'],'rounding_examples':rows},indent=2))

if __name__=='__main__':main()
