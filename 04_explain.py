import joblib,json,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt,shap
from sklearn.metrics import precision_recall_curve,confusion_matrix
Xtr,Xva,Xte,ytr,yva,yte,sc=joblib.load('data/split.joblib')
probs=joblib.load('results/test_probs.joblib');thr=json.load(open('models/thresholds.json'))
cmp=pd.read_csv('results/model_comparison_test.csv');imb=pd.read_csv('results/imbalance_val.csv')
# 1 PR curves
plt.figure(figsize=(6,4.5))
for n,p in probs.items():
    pr,rc,_=precision_recall_curve(yte,p);ap=cmp.set_index('model').loc[n,'pr_auc'];plt.plot(rc,pr,label=f'{n} (PR-AUC {ap:.2f})')
plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Precision-Recall curves (test set)');plt.legend(fontsize=8);plt.grid(alpha=.3);plt.tight_layout();plt.savefig('figures/pr_curves.png',dpi=160);plt.close()
# 2 confusion matrix xgb
p=probs['XGBoost'];yh=(p>=thr['XGBoost']).astype(int);cm=confusion_matrix(yte,yh)
fig,ax=plt.subplots(figsize=(4.2,3.8));ax.imshow(cm,cmap='Blues')
for i in range(2):
    for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=13)
ax.set_xticks([0,1],['Pred legit','Pred fraud']);ax.set_yticks([0,1],['Actual legit','Actual fraud']);ax.set_title('XGBoost confusion matrix (test)');plt.tight_layout();plt.savefig('figures/confusion_xgboost.png',dpi=160);plt.close()
# 3 imbalance chart
pv=imb.pivot(index='strategy',columns='model',values='pr_auc').loc[['Baseline','Class weights','SMOTE','Undersampling']]
pv.plot(kind='bar',figsize=(6,4));plt.ylabel('Validation PR-AUC');plt.title('Imbalance handling strategies');plt.xticks(rotation=0);plt.grid(axis='y',alpha=.3);plt.tight_layout();plt.savefig('figures/imbalance_strategies.png',dpi=160);plt.close()
# 4 model comparison
c=cmp.set_index('model')[['precision','recall','f1','pr_auc']];c.plot(kind='bar',figsize=(7,4));plt.xticks(rotation=15);plt.title('Model comparison (test set)');plt.grid(axis='y',alpha=.3);plt.legend(fontsize=8);plt.tight_layout();plt.savefig('figures/model_comparison.png',dpi=160);plt.close()
# 5 SHAP
m=joblib.load('models/xgboost.joblib');rng=np.random.RandomState(0)
idx=np.concatenate([np.where(yte.values==1)[0],rng.choice(np.where(yte.values==0)[0],1500,replace=False)])
S=Xte.iloc[idx];sv=shap.TreeExplainer(m).shap_values(S)
plt.figure();shap.summary_plot(sv,S,show=False,max_display=12);plt.tight_layout();plt.savefig('figures/shap_summary.png',dpi=160,bbox_inches='tight');plt.close()
plt.figure();shap.summary_plot(sv,S,plot_type='bar',show=False,max_display=12);plt.tight_layout();plt.savefig('figures/shap_bar.png',dpi=160,bbox_inches='tight');plt.close()
imp=pd.Series(np.abs(sv).mean(0),index=S.columns).sort_values(ascending=False);imp.head(10).to_csv('results/shap_top10.csv')
print('SHAP top10');print(imp.head(10).round(3))
# error analysis
amt_raw=sc.inverse_transform(Xte[['Time','Amount']])[:,1]
df=pd.DataFrame({'y':yte.values,'p':p,'pred':yh,'amount':amt_raw})
fn=df[(df.y==1)&(df.pred==0)];tp=df[(df.y==1)&(df.pred==1)];fp=df[(df.y==0)&(df.pred==1)]
print('FN',len(fn),'mean prob',fn.p.mean().round(3),'median amount',fn.amount.median().round(2),'| TP median amount',tp.amount.median().round(2),'| FP',len(fp),'median amount',fp.amount.median().round(2))
print('FN probs',sorted(fn.p.round(3).tolist()))
print('Total fraud amount test',round(df[df.y==1].amount.sum(),2),'caught',round(tp.amount.sum(),2),'missed',round(fn.amount.sum(),2))
