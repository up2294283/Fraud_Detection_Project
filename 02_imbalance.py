import joblib,json,time
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from common import *
Xtr,Xva,Xte,ytr,yva,yte,sc=joblib.load('data/split.joblib')
spw=(ytr==0).sum()/(ytr==1).sum()
def mk(name,cw):
    if name=='LogReg': return LogisticRegression(max_iter=2000,class_weight='balanced' if cw else None)
    return XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,tree_method='hist',n_jobs=1,eval_metric='aucpr',scale_pos_weight=spw if cw else 1,random_state=42)
res=[]
for name in ['LogReg','XGBoost']:
    for strat in ['Baseline','Class weights','SMOTE','Undersampling']:
        t=time.time();X,y=Xtr,ytr
        if strat=='SMOTE': X,y=SMOTE(random_state=42).fit_resample(Xtr,ytr)
        if strat=='Undersampling': X,y=RandomUnderSampler(random_state=42).fit_resample(Xtr,ytr)
        m=mk(name,strat=='Class weights').fit(X,y);p=m.predict_proba(Xva)[:,1]
        thr=best_thr(yva,p);r=metrics(yva,p,thr);r.update(model=name,strategy=strat,secs=round(time.time()-t,1))
        res.append(r);print(name,strat,{k:round(v,3) if isinstance(v,float) else v for k,v in r.items() if k in('precision','recall','f1','pr_auc','secs')},flush=True)
pd.DataFrame(res).to_csv('results/imbalance_val.csv',index=False)
