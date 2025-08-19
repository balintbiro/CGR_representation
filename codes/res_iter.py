import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import side_groups,bonds,structure,CGRepresentation,FrequencyCGR

sequences=pd.read_csv("data/ocr.csv")

sigro_coords=sequences["sequence"].sample(20).apply(CGRepresentation,args=(side_groups,))
struc_coords=sequences["sequence"].sample(20).apply(CGRepresentation,args=(structure,))
bond_coords=sequences["sequence"].sample(20).apply(CGRepresentation,args=(bonds,))

pd.DataFrame(columns=["roc_auc","dataset","resolution"]).to_csv("data/res_iter.csv")

resolutions=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for res in resolutions:
    sigro_mtxs=pd.DataFrame(normalize(sigro_coords.apply(FrequencyCGR,args=(res,)).tolist()))
    struc_mtxs=pd.DataFrame(normalize(struc_coords.apply(FrequencyCGR,args=(res,)).tolist()))
    bond_mtxs=pd.DataFrame(normalize(bond_coords.apply(FrequencyCGR,args=(res,)).tolist()))

    bond_mtxs["label"]=sequences["label"].values
    struc_mtxs["label"]=sequences["label"].values
    sigro_mtxs["label"]=sequences["label"].values

    skf=StratifiedKFold(n_splits=100,random_state=0,shuffle=True)

    results=pd.concat(
        [tester(mtx=bond_mtxs,dataset_name="bond_tata",skf=skf),
        tester(mtx=struc_mtxs,dataset_name="struc_tata",skf=skf),
        tester(mtx=sigro_mtxs,dataset_name="sigro_tata",skf=skf)]
    )
    results.columns=["roc_auc","dataset"]
    results["resolution"]=res
    results.to_csv("data/res_iter.csv",mode='a',index=False,header=False)
    print(f"{res} resolution is done!")
