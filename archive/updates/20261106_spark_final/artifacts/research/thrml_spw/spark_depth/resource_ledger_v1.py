"""Exact representation/reconstruction ledger for the active GPT2 protocol."""
from pathlib import Path
import json,math
from .prepare_v1 import OUT,sha

def main():
    root=OUT/'resource_ledger_v1';root.mkdir(exist_ok=False);experiment=OUT/'benchmark_v1';protocol=json.loads((experiment/'protocol.json').read_text())
    canonical_path=Path('results/thrml_spw/broad_verification_20261105/remote_metadata/additional_parameter_packages_v1/gpt2/package.json');canonical=json.loads(canonical_path.read_text())['canonical_parameters']
    n=sum(math.prod(p['shape'])for p in canonical);groups=sum(math.ceil(math.prod(p['shape'])/128)for p in canonical)
    assert n==124439808 and groups==972186
    comparisons=[]
    for name in ['selected_seed2026110801','matched_flat_seed2026110801']:
        path=experiment/name/'parameters.json';point=json.loads(path.read_text());rows=point['records']
        assert [r['name']for r in rows]==[r['name']for r in canonical]
        assert sum(r['parameters']for r in rows)==n and sum(r['range_groups']for r in rows)==groups
        assert all(r['samples']==32768 for r in rows)
        obs=sum(r['parameters']*r['samples']for r in rows)
        comparisons.append({'case':name,'parameters_file_sha256':sha(path),'logical_pbits':n,'observations_per_model_reconstruction':obs})
    assert comparisons[0]['observations_per_model_reconstruction']==comparisons[1]['observations_per_model_reconstruction']
    bitcounts={'BF16_weight_bits':16*n,'ideal_12bit_controls':12*n,'binary_states':n,'FP32_range_pairs':64*groups,'orientation_and_T_policy_upper_bound':17*len(canonical),'actual_uint16_control_array_bits':16*n}
    packed=bitcounts['ideal_12bit_controls']+bitcounts['binary_states']+bitcounts['FP32_range_pairs']+bitcounts['orientation_and_T_policy_upper_bound']
    unpacked=bitcounts['actual_uint16_control_array_bits']+bitcounts['binary_states']+bitcounts['FP32_range_pairs']+bitcounts['orientation_and_T_policy_upper_bound']
    obs=n*32768
    result={'source_sha256':sha(__file__),'benchmark_protocol_sha256':sha(experiment/'protocol.json'),'canonical_package_sha256':sha(canonical_path),
        'unique_parameters':n,'groups':groups,'logical_pbits_if_resident':n,'samples':32768,'comparisons':comparisons,'bits':bitcounts,
        'ideal_control_range_state_policy_bytes_rounded_up':math.ceil(packed/8),'ideal_fraction_of_BF16_weight_storage':packed/(16*n),
        'uint16_control_range_state_policy_bytes_rounded_up':math.ceil(unpacked/8),'uint16_fraction_of_BF16_weight_storage':unpacked/(16*n),
        'additional_BF16_working_weights_bytes_in_R1':2*n,'additional_original_BF16_CPU_copy_bytes_in_runner':2*n,
        'R1_observations_amortized_over_this_586_window_evaluation':obs/586,'R1_observations_amortized_per_scored_target_in_this_evaluation':obs/283225,
        'fresh_full_parameter_reconstruction_per_window_hypothesis_observations':obs*586,
        'notes':['Idealbitcountsarenotphysicalsiliconarea. Actualsimulatoralsouses temporarycounters,arrays,modelactivationsandframeworkmetadata.',
            'CurrentR1weightpointisreusedacrossentireevaluation; pertargetamortizationisnotdirectstochasticexecutioncost.',
            'Freshfullreconstructionperwindowisahypotheticaldifferentexecutionpolicy,notmeasuredR2.',
            'Nativehardwarelatency/energyrequiresdeviceupdatefrequency,parallelcapacity,biasretention,readout,operatorreuseandcalibrationcosts. Noabsoluteclaimcomputed.']}
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['unique_parameters','groups','ideal_fraction_of_BF16_weight_storage','uint16_fraction_of_BF16_weight_storage','R1_observations_amortized_per_scored_target_in_this_evaluation']}))

if __name__=='__main__':main()
