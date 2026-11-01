import joblib,json,time,itertools,os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from common import *
Xtr,Xva,Xte,ytr,yva,yte,sc=joblib.load('data/split.joblib')
spw=(ytr==0).sum()/(ytr==1).sum()
rows=[];probs={};best={}
def run(name,cands,build,fit=lambda m:m.fit(Xtr,ytr),pred=lambda m,X:m.predict_proba(X)[:,1]):
    t=time.time();bs=None
    for c in cands:
        m=fit(build(c)) ;p=pred(m,Xva);s=average_precision_score(yva,p)
        print(name,c,'val PR-AUC',round(s,4),flush=True)
        if bs is None or s>bs[0]:bs=(s,c,m)
    s,c,m=bs;pv=pred(m,Xva);thr=best_thr(yva,pv);pt=pred(m,Xte)
    r=metrics(yte,pt,thr);r.update(model=name,params=str(c),val_pr_auc=s,secs=round(time.time()-t,1));rows.append(r);probs[name]=pt;best[name]=(m,thr)
    print('>>',name,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
run('Logistic Regression',[dict(C=c) for c in (0.01,0.1,1,10)],lambda c:LogisticRegression(max_iter=3000,class_weight='balanced',**c))
run('Random Forest',[dict(n_estimators=150,max_depth=d,min_samples_leaf=l) for d,l in ((10,2),(16,1))],
    lambda c:RandomForestClassifier(class_weight='balanced_subsample',n_jobs=1,random_state=42,max_features='sqrt',**c))
run('XGBoost',[dict(max_depth=d,learning_rate=lr,n_estimators=n,subsample=0.8,colsample_bytree=0.8) for d,lr,n in ((4,0.1,300),(6,0.05,400),(5,0.1,300))],
    lambda c:XGBClassifier(tree_method='hist',n_jobs=1,eval_metric='aucpr',random_state=42,**c))
# neural network
import tensorflow as tf
from tensorflow import keras
def build_nn(c):
    tf.random.set_seed(42);m=keras.Sequential([keras.layers.Input(shape=(30,)),keras.layers.Dense(c['u'],activation='relu'),keras.layers.Dropout(c['d']),keras.layers.Dense(c['u']//2,activation='relu'),keras.layers.Dropout(c['d']),keras.layers.Dense(1,activation='sigmoid')])
    m.compile(optimizer=keras.optimizers.Adam(1e-3),loss='binary_crossentropy');return m
def fit_nn(m):
    m.fit(Xtr.values,ytr.values,validation_data=(Xva.values,yva.values),epochs=15,batch_size=2048,class_weight={0:1.0,1:float(np.sqrt(spw))},
          callbacks=[keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor='val_loss')],verbose=0);return m
run('Neural Network',[dict(u=64,d=0.2),dict(u=32,d=0.3)],build_nn,fit_nn,lambda m,X:m.predict(X.values,verbose=0).ravel())
df=pd.DataFrame(rows);df.to_csv('results/model_comparison_test.csv',index=False)
joblib.dump(probs,'results/test_probs.joblib')
for n,(m,thr) in best.items():
    if n=='Neural Network': m.save('models/nn.keras')
    else: joblib.dump(m,'models/'+n.replace(' ','_').lower()+'.joblib')
json.dump({n:b[1] for n,b in best.items()},open('models/thresholds.json','w'))
print(df[['model','precision','recall','f1','pr_auc','roc_auc','tp','fp','fn']].round(3).to_string())
