import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix,precision_recall_curve
def best_thr(y,p):
    pr,rc,th=precision_recall_curve(y,p)
    f=2*pr[:-1]*rc[:-1]/np.maximum(pr[:-1]+rc[:-1],1e-12)
    i=int(np.argmax(f));return float(th[i])
def metrics(y,p,thr=0.5):
    yh=(p>=thr).astype(int);tn,fp,fn,tp=confusion_matrix(y,yh).ravel()
    return dict(precision=precision_score(y,yh,zero_division=0),recall=recall_score(y,yh),f1=f1_score(y,yh),
        pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),thr=thr,tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
