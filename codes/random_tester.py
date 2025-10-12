import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import monomer_groupings,CGRepresentation,FrequencyCGR

sequences=pd.read_csv("data/deeploc_clean.csv")
sample=pd.concat([
    sequences[sequences["label"]==0].sample(n=500,random_state=0),
    sequences[sequences["label"]==1].sample(n=500,random_state=0)
]).reset_index(drop=True)
X,y=sample["sequence"],sample["label"]
X_train, X_test, y_train, y_test=train_test_split(X,y)
train_idx,test_idx=X_train.index,X_test.index

pd.DataFrame(columns=["order","accuracy"]).to_csv("data/random_grouping.csv",index=False)

def random_grouping()->tuple:
    groupings=pd.DataFrame(monomer_groupings["protein"]["polar"])
    new_order=np.random.choice(
        a=groupings.columns,
        size=len(groupings.columns),
        replace=False
    )
    groupings.columns=new_order
    random_grouping=groupings.to_dict(orient="list")
    return new_order,random_grouping

for i in range(2):
    aa_list,current_grouping=random_grouping()
    cgr=sample["sequence"].apply(CGRepresentation,args=(current_grouping,0.865,))
    fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
    
    X_train, X_test, y_train, y_test=(
        fcgr.iloc[train_idx],
        fcgr.iloc[test_idx],
        sample["label"].iloc[train_idx],
        sample["label"].iloc[test_idx]
    )

    clf=XGBClassifier()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    acc=accuracy_score(y_true=y_test,y_pred=y_pred)
    pd.DataFrame([["".join(aa_list),acc]]).to_csv("data/random_grouping.csv",mode='a',index=False,header=False)
