import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold,cross_validate

from utils import monomer_groupings,CGRepresentation,FrequencyCGR,tester

sequences=pd.read_csv("data/dros_enhancer.csv")

pd.DataFrame(columns=["test_accuracy","test_roc_auc","dataset","resolution","scaling_factor"]).to_csv("data/res_iter_dna.csv",index=False)

resolutions=np.geomspace(start=2,stop=50,num=10,dtype=int)[::-1]
scaling_factors=np.linspace(start=0.1,stop=0.7,num=10)[::-1]

encode=monomer_groupings["dna"]

for res in resolutions:
    for sf in scaling_factors:
        sigro_coords=sequences["sequence"].apply(CGRepresentation,args=(encode["side_groups"],sf,))
        struc_coords=sequences["sequence"].apply(CGRepresentation,args=(encode["structure"],sf,))
        bond_coords=sequences["sequence"].apply(CGRepresentation,args=(encode["bonds"],sf,))

        sigro_mtxs=pd.DataFrame(normalize(sigro_coords.apply(FrequencyCGR,args=(res,)).tolist()))
        struc_mtxs=pd.DataFrame(normalize(struc_coords.apply(FrequencyCGR,args=(res,)).tolist()))
        bond_mtxs=pd.DataFrame(normalize(bond_coords.apply(FrequencyCGR,args=(res,)).tolist()))

        bond_mtxs["label"]=sequences["label"].values
        struc_mtxs["label"]=sequences["label"].values
        sigro_mtxs["label"]=sequences["label"].values

        skf=StratifiedKFold(n_splits=20,random_state=0,shuffle=True)

        bond_results=tester(mtx=bond_mtxs,dataset_name="bond",skf=skf,resolution=res,scaling_factor=sf)
        bond_results.to_csv("data/res_iter_dna.csv",mode='a',index=False,header=False)

        struc_results=tester(mtx=struc_mtxs,dataset_name="struc",skf=skf,resolution=res,scaling_factor=sf)
        struc_results.to_csv("data/res_iter_dna.csv",mode='a',index=False,header=False)

        sigro_results=tester(mtx=sigro_mtxs,dataset_name="sigro",skf=skf,resolution=res,scaling_factor=sf)
        sigro_results.to_csv("data/res_iter_dna.csv",mode='a',index=False,header=False)
        print(f"{res} resolution is done!")
