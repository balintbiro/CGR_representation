import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import side_groups,bonds,structure,CGRepresentation,FrequencyCGR,tester

sequences=pd.read_csv("data/ocr.csv").sample(200)

pd.DataFrame(columns=["accuracy","dataset","resolution","scaling_factor"]).to_csv("data/res_iter.csv",index=False)

resolutions=[8,10,16,32]
scaling_factors=np.linspace(start=0.1,stop=0.5,num=5)

for res in resolutions:
    for sf in scaling_factors:
        sigro_coords=sequences["sequence"].apply(CGRepresentation,args=(side_groups,sf,))
        struc_coords=sequences["sequence"].apply(CGRepresentation,args=(structure,sf,))
        bond_coords=sequences["sequence"].apply(CGRepresentation,args=(bonds,sf,))

        sigro_mtxs=pd.DataFrame(normalize(sigro_coords.apply(FrequencyCGR,args=(res,)).tolist()))
        struc_mtxs=pd.DataFrame(normalize(struc_coords.apply(FrequencyCGR,args=(res,)).tolist()))
        bond_mtxs=pd.DataFrame(normalize(bond_coords.apply(FrequencyCGR,args=(res,)).tolist()))

        bond_mtxs["label"]=sequences["label"].values
        struc_mtxs["label"]=sequences["label"].values
        sigro_mtxs["label"]=sequences["label"].values

        skf=StratifiedKFold(n_splits=3,random_state=0,shuffle=True)

        bond_results=tester(mtx=bond_mtxs,dataset_name="bond_ocr",skf=skf)
        bond_results.columns=["accuracy","dataset"]
        bond_results["resolution"]=res
        bond_results["scaling_factor"]=sf
        bond_results.to_csv("data/res_iter.csv",mode='a',index=False,header=False)

        struc_results=tester(mtx=struc_mtxs,dataset_name="struc_ocr",skf=skf)
        struc_results.columns=["accuracy","dataset"]
        struc_results["resolution"]=res
        struc_results["scaling_factor"]=sf
        struc_results.to_csv("data/res_iter.csv",mode='a',index=False,header=False)

        sigro_results=tester(mtx=sigro_mtxs,dataset_name="sigro_ocr",skf=skf)
        sigro_results.columns=["accuracy","dataset"]
        sigro_results["resolution"]=res
        sigro_results["scaling_factor"]=sf
        sigro_results.to_csv("data/res_iter.csv",mode='a',index=False,header=False)

        print(f"{res} resolution is done!")
