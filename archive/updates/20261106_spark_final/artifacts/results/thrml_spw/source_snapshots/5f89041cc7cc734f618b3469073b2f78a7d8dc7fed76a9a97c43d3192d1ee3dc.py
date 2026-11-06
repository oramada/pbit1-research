"""Exact known representation counts, separated from unmeasured hardware costs."""
import json,math
from .prepare_v1 import OUT,sha


def main():
    rows=[]
    for model,folder in [('GPT2','benchmark_v1'),('DistilBERT','distilbert_benchmark_v1'),('LFM3','lfm3_benchmark_v2'),('GPT2 lower-work','reduced_work_gpu_v1')]:
        root=OUT/folder;result=json.loads((root/'results.json').read_text());case=next(x for x in result['cases']if x['case']['kind']in ['selected','reduced'])
        parameter_file=root/case['case']['name']/'parameters.json';p=json.loads(parameter_file.read_text());records=p.get('records',p.get('parameters'))
        N=sum(x['parameters']for x in records);G=sum(x['range_groups']for x in records);updates=sum(x['parameters']*x['samples']for x in records)
        rows.append({'model':model,'source_parameter_sha256':sha(parameter_file),'unique_parameters':N,'range_groups':G,'logical_binary_states':N,
          'ideal_12bit_command_bits':12*N,'software_uint16_command_array_bytes':2*N,'FP32_range_metadata_bytes':8*G,
          'ideal_packed_commands_plus_ranges_bytes':math.ceil(12*N/8)+8*G,'binary_state_bits_additional':N,
          'ideal_commands_ranges_states_over_BF16':(13*N+64*G)/(16*N),
          'uint16_commands_ranges_states_over_BF16':(17*N+64*G)/(16*N),
          'BF16_decoded_working_weights_bytes_additional':2*N,'original_BF16_reference_bytes_if_retained_additional':2*N,
          'nominal_independent_Bernoulli_observations_per_reconstruction':updates,'actual_binomial_count_variates_per_reconstruction':N,
          'maximum_samples':max(x['samples']for x in records),'maximum_saved_GPU_allocation_bytes':max(x['GPU_peak_allocated_bytes']for x in result['cases'])})
    output={'source_sha256':sha(__file__),'rows':rows,
      'scope':'Exact declared array/state counts, not actual chip area or full software RSS. Policy/shape/addressing metadata, framework objects and transient tensors are additional. Conceptual physical binary-state bits are not claimed to be allocated as packed software arrays. Ideal12bit packing is not implemented by uint16 program arrays.',
      'hardware_costs_not_identified':['energy per free-node update','clamped-node holding/leakage power','bias-programming and calibration energy','readout and accumulation energy','control-memory/circuit area','physical sampling frequency and correlation','finite-device-pool scheduling latency'],
      'execution_boundary':'R1 reconstructs a weight matrix and reuses it across task inputs; decoded BF16 working weights remain. Direct stochastic execution pays sampling/readout/accumulation per operator. Large-model binomial simulation does not explicitly emit every nominal Bernoulli observation. Native THRML cases are reported separately.'}
    (OUT/'resource_ledger_final_v1.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'models':len(rows),'hardware_energy_or_speedup_claimed':False}))
if __name__=='__main__':main()
