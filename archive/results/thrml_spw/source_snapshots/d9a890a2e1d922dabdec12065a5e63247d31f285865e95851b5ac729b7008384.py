"""Unchanged stochastic producer followed by versioned H100-control auditor."""
import argparse,subprocess,sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--budget',type=int,required=True);p.add_argument('--bits',type=int,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();args=['--arm','mean','--budget',str(a.budget),'--word-bits',str(a.bits),'--seed',str(a.seed)]
    for module in ['modal_h100_word_v2','verify_modal_h100_word_v3']:
        subprocess.run([sys.executable,'-m','research.thrml_spw.lfm3.'+module,*args],check=True)
if __name__=='__main__':main()
