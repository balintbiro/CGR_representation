import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_predict,LeaveOneOut
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import side_groups,bonds,structure,CGRepresentation,FrequencyCGR,tester

sequences=pd.read_csv("data/dros_enhancer.csv")

sigro_coords=sequences["sequence"].apply(CGRepresentation,args=(side_groups,))
struc_coords=sequences["sequence"].apply(CGRepresentation,args=(structure,))
bond_coords=sequences["sequence"].apply(CGRepresentation,args=(bonds,))

pd.DataFrame(columns=["prob0","prob1","resolution","dataset","y_true"]).to_csv("data/res_iter.csv",index=False)

resolutions=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for res in resolutions:
    sigro_mtxs=pd.DataFrame(normalize(sigro_coords.apply(FrequencyCGR,args=(res,)).tolist()))
    struc_mtxs=pd.DataFrame(normalize(struc_coords.apply(FrequencyCGR,args=(res,)).tolist()))
    bond_mtxs=pd.DataFrame(normalize(bond_coords.apply(FrequencyCGR,args=(res,)).tolist()))

    bond_mtxs["label"]=sequences["label"].values
    struc_mtxs["label"]=sequences["label"].values
    sigro_mtxs["label"]=sequences["label"].values

    clf=XGBClassifier(random_state=0)
    loocv=LeaveOneOut()

    # SIde Groups
    sigro_probs=pd.DataFrame(
        data=cross_val_predict(estimator=clf,X=sigro_mtxs.drop(columns=["label"]),y=sigro_mtxs["label"],cv=loocv,method="predict_proba"),
        columns=["prob0","prob1"]
    )
    sigro_probs["Resolution"]=res
    sigro_probs["Dataset"]="side_group"
    sigro_probs["y_true"]=sequences["label"]
    sigro_probs.to_csv("data/res_iter.csv",mode='a',index=False,header=False)

    # H-bond strengths
    bond_probs=pd.DataFrame(
        data=cross_val_predict(estimator=clf,X=bond_mtxs.drop(columns=["label"]),y=bond_mtxs["label"],cv=loocv,method="predict_proba"),
        columns=["prob0","prob1"]
    )
    bond_probs["Resolution"]=res
    bond_probs["Dataset"]="bond"
    bond_probs["y_true"]=sequences["label"]
    bond_probs.to_csv("data/res_iter.csv",mode='a',index=False,header=False)

    # Structure
    struc_probs=pd.DataFrame(
        data=cross_val_predict(estimator=clf,X=struc_mtxs.drop(columns=["label"]),y=struc_mtxs["label"],cv=loocv,method="predict_proba"),
        columns=["prob0","prob1"]
    )
    struc_probs["Resolution"]=res
    struc_probs["Dataset"]="structure"
    struc_probs["y_true"]=sequences["label"]
    struc_probs.to_csv("data/res_iter.csv",mode='a',index=False,header=False)
    print(f"{res} resolution is done!")
