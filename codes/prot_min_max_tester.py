import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import monomer_groupings,CGRepresentation,FrequencyCGR,aa_grouping,tester

sequences=pd.read_csv("data/deeploc_clean.csv")

random_grouping_results=pd.read_csv("data/random_grouping.csv",nrows=10_000)

worst=random_grouping_results[random_grouping_results.accuracy==random_grouping_results.accuracy.min()]["order"].values[0]
best=random_grouping_results[random_grouping_results.accuracy==random_grouping_results.accuracy.max()]["order"].values[0]

min_max_order={}

aa_grouping(all_groupings=min_max_order,grouping_name="min",first_group=list(worst))
aa_grouping(all_groupings=min_max_order,grouping_name="max",first_group=list(best))

pd.DataFrame(columns=["test_accuracy","dataset","resolution","scaling_factor"]).to_csv("data/prot_min_max_results.csv",index=False)

min_cgr=sequences["sequence"].apply(CGRepresentation,args=(min_max_order["min"],0.856,))
max_cgr=sequences["sequence"].apply(CGRepresentation,args=(min_max_order["max"],0.856,))

min_fcgr=pd.DataFrame(normalize(min_cgr.apply(FrequencyCGR,args=(40,)).tolist()))
max_fcgr=pd.DataFrame(normalize(max_cgr.apply(FrequencyCGR,args=(40,)).tolist()))

min_fcgr["label"]=sequences["label"].values
max_fcgr["label"]=sequences["label"].values

skf=StratifiedKFold(n_splits=20,random_state=0,shuffle=True)

min_results=tester(mtx=min_fcgr,dataset_name="min",skf=skf,resolution=40,scaling_factor=0.856)
min_results["resolution"]=40
min_results["scaling_factor"]=0.856
min_results.to_csv("data/prot_min_max_results.csv",mode='a',index=False,header=False)

max_results=tester(mtx=max_fcgr,dataset_name="max",skf=skf,resolution=40,scaling_factor=0.856)
max_results["resolution"]=40
max_results["scaling_factor"]=0.856
max_results.to_csv("data/prot_min_max_results.csv",mode='a',index=False,header=False)
