import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
df=pd.read_csv('data/creditcard.csv')
print('raw',df.shape,'dups',df.duplicated().sum(),'nulls',df.isna().sum().sum())
df=df.drop_duplicates().reset_index(drop=True)
print('clean',df.shape,'fraud',df.Class.sum(),f'{df.Class.mean()*100:.3f}%')
X=df.drop(columns='Class');y=df.Class
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,stratify=y,random_state=42)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.50,stratify=ytmp,random_state=42)
sc=RobustScaler().fit(Xtr[['Time','Amount']])
def tr(a):
    a=a.copy();a[['Time','Amount']]=sc.transform(a[['Time','Amount']]);return a
Xtr,Xva,Xte=tr(Xtr),tr(Xva),tr(Xte)
for n,a,b in [('train',Xtr,ytr),('val',Xva,yva),('test',Xte,yte)]:
    print(n,a.shape,'fraud',int(b.sum()))
joblib.dump((Xtr,Xva,Xte,ytr,yva,yte,sc),'data/split.joblib')
joblib.dump(list(X.columns),'models/features.joblib')
