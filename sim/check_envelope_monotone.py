import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from certified import certified_envelope
from kgw import h_kgw
TS=np.concatenate([[0.001,0.005],np.linspace(0.01,0.99,99)])
for scheme in ('gumbel','kgw'):
  for d in (0.1,0.2):
    H={}
    for n in range(50,401):
      H[n]= certified_envelope(n,0.3,d,TS) if scheme=='gumbel' else np.maximum(h_kgw(TS,n,0.5,0.3,d,2.0,V=None)[0]-1e-9,0)
    viol=[]
    for lo,hi in ((50,100),(100,200),(200,401)):
      for n in range(lo,hi):
        dif=(H[lo]-H[n]).max()
        if dif>1e-9: viol.append((lo,n,round(float(dif),5)))
    adj=[(n,round(float((H[n]-H[n+1]).max()),5)) for n in range(50,400) if (H[n]-H[n+1]).max()>1e-9]
    print(scheme,d,'stratum-floor violations:',len(viol),viol[:5],' adjacent non-monotone n:',len(adj),adj[:5])
