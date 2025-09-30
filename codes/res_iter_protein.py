import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold,cross_validate

from utils import monomer_groupings,CGRepresentation,FrequencyCGR

sequences=pd.read_csv("data/deeploc_clean.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["aliphatic"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X,y=fcgr,sequences["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y)
train_idx,test_idx=X_train.index,X_test.index
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"Aliphatic acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["size1"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"Size1 acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["size2"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"size2 acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["hydrophobic"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"hydrophobic acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["positive"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"positive acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["charged"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"charged acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["polar"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"polar acc: {accuracy_score(y_test,clf.predict(X_test))}")

cgr=sequences["sequence"].apply(CGRepresentation,args=(monomer_groupings["protein"]["aromatic"],0.5,))
fcgr=pd.DataFrame(normalize(cgr.apply(FrequencyCGR,args=(40,)).tolist()))
X_train, X_test, y_train, y_test = fcgr.iloc[train_idx],fcgr.iloc[test_idx],sequences["label"].iloc[train_idx],sequences["label"].iloc[test_idx]
clf=XGBClassifier(random_state=0)
clf.fit(X_train,y_train)
print(f"aromatic acc: {accuracy_score(y_test,clf.predict(X_test))}")
