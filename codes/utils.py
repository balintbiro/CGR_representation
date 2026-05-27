
import requests
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

side_groups={
    'A':[-1,1],
    'C':[-1,-1],
    'G':[1,-1],
    'T':[1,1]
}
structure={
    'A':[-1,1],
    'C':[1,-1],
    'G':[-1,-1],
    'T':[1,1]
}
bonds={
    'A':[-1,1],
    'C':[1,1],
    'G':[1,-1],
    'T':[-1,-1]
}

def tester(mtx:pd.DataFrame,dataset_name:str,skf:StratifiedKFold)->pd.DataFrame:
    """
    Performs Stratified K fold Cross Validation on a FCGR matrix.

    Parameters:
    - mtx: matrix of bonds, sigro or struc
    - dataset_name: bond, sigro or struc
    - skf: instantiated StratifiedKFold

    Returns:
    - dataframe of the specfied matric and dataset name
    """
    X,y=mtx.drop(columns=["label"]).values,mtx["label"].values
    results=[]
    for i, (train_index,test_index) in enumerate(skf.split(X,y)):
        clf=LogisticRegression(random_state=0,n_jobs=-1)
        clf.fit(X[train_index],y[train_index])
        y_pred=clf.predict(X[test_index])
        results.append(accuracy_score(y_true=y[test_index],y_pred=y_pred))
    return pd.DataFrame(
        data=[results,len(results)*[dataset_name]],
    ).T

def loggerConfig(logfile:str)->None:
    """
    Setting up the logging structure.

    Parameters:
    - logfile:str, filepath of the logfile

    Returns:
    - None
    """
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        filemode='a',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10 * 16 * 16, 1)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class DeepLoc:
    def __init__(self):
        self.url="https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"

    def get(self,outfile:str)->None:
        response=requests.