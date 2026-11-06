"""Exact finite-budget allocation for a frozen inverse-T loss surrogate."""
import numpy as np

def allocate(coefficients,weight_units,budget,levels=(1,2,4,8,16)):
    a=np.asarray(coefficients,float);units=np.asarray(weight_units,int)
    if a.ndim!=1 or len(a)!=len(units) or np.any(~np.isfinite(a)) or np.any(a<=0) or np.any(units<=0):raise ValueError('Positive finite coefficients and integer weights required')
    dp={0:(0.,())}
    for coefficient,unit in zip(a,units):
        nxt={}
        for used,(cost,path) in dp.items():
            for level in levels:
                total=used+int(unit)*int(level)
                if total>budget:continue
                candidate=(cost+float(coefficient)/int(level),path+(int(level),))
                if total not in nxt or candidate<nxt[total]:nxt[total]=candidate
        dp=nxt
    if budget not in dp:raise ValueError('Exact budget infeasible')
    cost,path=dp[budget];return {'levels':list(path),'objective':cost,'budget_used':sum(int(u)*t for u,t in zip(units,path))}
