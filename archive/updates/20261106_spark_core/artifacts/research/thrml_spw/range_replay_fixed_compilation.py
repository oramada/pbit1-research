"""Independent cold-process pilot with timing-based XLA autotuning disabled."""
import os,sys,argparse
from . import range_replay_diagnostic as diagnostic
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicate',type=int,choices=[0,1],required=True);args=parser.parse_args()
    assert os.environ.get('XLA_FLAGS')=='--xla_gpu_autotune_level=0'
    diagnostic.OUT=diagnostic.ROOT/'reproducibility_diagnostic'/f'autotune0_rep{args.replicate}'
    sys.argv=[sys.argv[0],'--mode','cold'];diagnostic.main()
if __name__=='__main__':main()
