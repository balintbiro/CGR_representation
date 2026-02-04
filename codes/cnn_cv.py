import os
import click
import torch
import logging
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from utils import loggerConfig,Cnn

device="cuda" if torch.cuda.is_available() else "cpu"

logger=logging.getLogger(__name__)

def cross_validate(X:pd.DataFrame|np.ndarray,y:pd.Series|np.array,n_split=3)->list:
    skf=StratifiedKFold(n_splits=n_split,random_state=0,shuffle=True)
    acc_scores=[]
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test=X[train_index],X[test_index],y[train_index],y[test_index]
        torch.manual_seed(0)

        cnn = NeuralNetBinaryClassifier(
            Cnn,
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
        )

        cnn.fit(X_train,y_train)
        y_pred=cnn.predict(X_test)
        acc_score=accuracy_score(y_true=y_test,y_pred=y_pred)
        acc_scores.append(acc_score)
    return acc_scores
