# import the necessary libraries
import os
import sys
import click
import torch
import random
import logging
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier,NeuralNetClassifier
from pathlib import Path

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from codes.utils import loggerConfig,Cnn,ResNet

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"

device="cuda" if torch.cuda.is_available() else "cpu"

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def cross_validate(X:pd.DataFrame|np.ndarray,y:pd.Series|np.ndarray,task:str,model:str,n_split=3)->list:
    """
    Perform cross validation on the given data.

    Parameters:
    - X: DataFrame or numpy array containing the features.
    - y: Series or numpy array containing the labels.
    - task: binary|multiclass
    - n_split: Number of splits for cross validation. 10 was used for the publication.

    Returns:
    - List of accuracy scores for each fold.
    """
    # seeding for reproducibility in the cross validation splits and the training and testing of the CNN
    seed=0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    skf=StratifiedKFold(n_splits=n_split,random_state=seed,shuffle=True)
    scores=[]
    if task=="binary":
        y=y.values.astype("float32")
    elif task=="multiclass":
        y=y.values.astype(np.int64)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test=X[train_index],X[test_index],y[train_index],y[test_index]
        seed=0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if task=="binary":
            if model=="custom":
                cnn=NeuralNetBinaryClassifier(
                    Cnn(output_dim=1),
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
            elif model=="resnet":
                cnn=NeuralNetBinaryClassifier(
                    ResNet(output_dim=1),
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
        elif task=="multiclass":
            if model=="custom":
                cnn=NeuralNetClassifier(
                    Cnn(output_dim=5),
                    criterion=torch.nn.CrossEntropyLoss,
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
            elif model=="resnet":
                cnn=NeuralNetClassifier(
                    ResNet(output_dim=5),
                    criterion=torch.nn.CrossEntropyLoss,
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
        cnn.fit(X_train,y_train)
        if task=="binary":
            auroc=roc_auc_score(y_true=y_test,y_score=cnn.predict_proba(X_test)[:,1])
        elif task=="multiclass":
            auroc=roc_auc_score(y_true=y_test,y_score=cnn.predict_proba(X_test),average="macro",multi_class="ovr")
        scores.append([model,auroc])
        if (i>0) and (i%5==0):
            logger.info("Completed %d/%d folds",i,n_split)
            logger.info(f"Used model is: {cnn}")
    return scores

# define the command line interface using click
@click.command()
@click.option(
    "--logfile",
    help="Path to logfile[.log]",
    required=True
)
@click.option(
    "--fcgr_matrix",
    help="Path to file[.csv] containing FCGRs",
    required=True
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the distances",
    required=True
)
@click.option(
    "--name",
    help="Name of the dataset that is used for CV, like Min, Q1 etc.",
    required=True
)
@click.option(
    "--model",
    help="Name of the model to use.",
    required=True,
    type=click.Choice(
        ["custom","resnet"],
        case_sensitive=False
    )
)
@click.option(
    "--task",
    help="Name of the task to perform",
    required=True,
    type=click.Choice(
        ["binary","multiclass"],
        case_sensitive=False
    )
)
@click.option(
    "--res",
    help="Resolution (int) of the FCGR",
    required=True,
    type=int
)
@click.option(
    "--rank",
    help="Rank of the dataset (min...max)",
    required=True,
    type=click.Choice(
        ["min","q1","q2","q3","max"],
        case_sensitive=False
    )
)
@click.option(
    "--n",
    help="Number of cross validation splits",
    required=True,
    type=int
)

def main(
        logfile,
        fcgr_matrix,
        outfile,
        name,
        model,
        task,
        res,
        rank,
        n
    )->None:
    """
    Perform CNN based cross validation on FCGR matrix.

    Parameters:
    - logfile: Path to logfile[.log]
    - fcgr_matrix: Path to file[.csv] containing FCGRs
    - outfile: Path to outfile[.csv] that will contain the accuracies
    - res: Resolution (int) of the FCGR
    - n: Number of cross validation splits

    Returns:
    - None
    """
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["model","auroc","dataset","rank"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    logger.info("Loading FCGR matrix from %s",fcgr_matrix)
    fcgr_df=pd.read_csv(fcgr_matrix)
    X,y=fcgr_df.drop(columns=["label"]).div(fcgr_df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32"),fcgr_df["label"]
    XCnn=X.reshape(-1,1,res,res)
    logger.info("Starting cross validation with %d splits",n)
    scores=cross_validate(X=XCnn,y=y,task=task,model=model,n_split=n)
    # save the scores in the output file
    scores_df=pd.DataFrame(scores)
    scores_df["dataset"]=name
    scores_df["rank"]=rank
    scores_df.to_csv(outfile,index=False,header=False,mode="a")
    logger.info("Scores saved to %s",outfile)

if __name__ == "__main__":
    main()