# import the necessary libraries
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

def cross_validate(X:pd.DataFrame|np.ndarray,y:pd.Series|np.ndarray,n_split=3)->list:
    """
    Perform cross validation on the given data.

    Parameters:
    - X: DataFrame or numpy array containing the features.
    - y: Series or numpy array containing the labels.
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
    acc_scores=[]
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test=X[train_index],X[test_index],y[train_index],y[test_index]

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
        if (i>0) and (i%5==0):
            logger.info("Completed %d/%d folds",i,n_split)
    return acc_scores

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
    "--res",
    help="Resolution (int) of the FCGR",
    required=True,
    type=int
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
        res,
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
        out_df=pd.DataFrame(columns=["Accuracy","Name"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    logger.info("Loading FCGR matrix from %s",fcgr_matrix)
    fcgr_df=pd.read_csv(fcgr_matrix)
    X,y=fcgr_df.drop(columns=["label"]).values.astype("float32"),fcgr_df["label"].values.astype("float32")
    XCnn=X.reshape(-1,1,res,res)
    logger.info("Starting cross validation with %d splits",n)
    acc_scores=cross_validate(X=XCnn,y=y,n_split=n)
    # save the accuracies in the output file
    acc_df=pd.DataFrame()
    acc_df["Accuracy"]=acc_scores
    acc_df["Name"]=name
    acc_df.to_csv(outfile,index=False,header=False,mode="a")
    logger.info("Accuracies saved to %s",outfile)

if __name__ == "__main__":
    main()