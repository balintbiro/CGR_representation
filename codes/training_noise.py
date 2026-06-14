# import the necessary libraries
import os
import sys
import click
import random
import logging
import datetime
import numpy as np
import pandas as pd
import subprocess
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier,NeuralNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score

from utils import loggerConfig,ResNet,Cnn

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

# add CLI arguments
@click.command()
@click.option(
    "--logfile",
    help="Path to logfile[.log]",
    required=True
)
@click.option(
    "--seqfile",
    help="Path to file[.csv] containing sequences and labels",
    required=True
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the accuracies accross encodings",
    required=True
)
@click.option(
    "--n-seed",
    help="Number of seed variations",
    required=True,
    type=int
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
    "--dataset_name",
    help="Name of the dataset to use",
    required=True,
    type=click.Choice(
        ["deeploc","tox","pfam","immune"],
        case_sensitive=False
    )
)

def main(
        logfile,
        seqfile,
        outfile,
        n_seed,
        task,
        dataset_name
    ):
    fcgrfile,sf,res="../data/temp_encoding.csv",0.865,35
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["encoding","auroc","accuracy","type"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    proteogenic_aas="ACDEFGHIKLMNPQRSTVWY"
    seed=0
    # set "random" (changing in every iterations) seed for encoding generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    encoding=''.join(np.random.choice(a=list(proteogenic_aas),size=len(proteogenic_aas),replace=False))

    # set seed for reproducible results in training and testing
    # creating the random encodings and the corresponding FCGRs
    subprocess.run(f"""Rscript --vanilla FCGR_gen.R --encoding {encoding} --output_file {fcgrfile} --input_filename {seqfile} --scaling_factor {sf} --resolution {res}""",shell=True)
    # getting the FCGRs and training the CNN on them
    df=pd.read_csv(fcgrfile)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for j in range(n_seed):
        torch.manual_seed(j)
        np.random.seed(j)
        random.seed(j)
    
        if task=="binary":
            custom=NeuralNetBinaryClassifier(
                Cnn(output_dim=1),
                max_epochs=10,
                lr=0.001,
                optimizer=torch.optim.Adam,
                device=device,
                train_split=None,
                iterator_train__shuffle=None
            )
            resnet=NeuralNetBinaryClassifier(
                ResNet(output_dim=1),
                max_epochs=10,
                lr=0.001,
                optimizer=torch.optim.Adam,
                device=device,
                train_split=None,
                iterator_train__shuffle=None
            )
            y=df["label"].values.astype("float32")
        elif task=="multiclass":
            custom=NeuralNetClassifier(
                Cnn(output_dim=5),
                criterion=torch.nn.CrossEntropyLoss,
                max_epochs=10,
                lr=0.001,
                optimizer=torch.optim.Adam,
                device=device,
                train_split=None,
                iterator_train__shuffle=None
            )
            resnet=NeuralNetClassifier(
                ResNet(output_dim=5),
                criterion=torch.nn.CrossEntropyLoss,
                max_epochs=10,
                lr=0.001,
                optimizer=torch.optim.Adam,
                device=device,
                train_split=None,
                iterator_train__shuffle=None
            )
            y=df["label"].values.astype(np.int64)
        X=df.drop(columns=["label"]).div(df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32")
        XCnn = X.reshape(-1, 1, res,res)
        XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=seed, stratify=y)
        custom.fit(XCnn_train, y_train)
        # get the accuracy scores on the testing data and save them in the output file
        if task=="binary":
            customf1=f1_score(y_true=y_test,y_pred=custom.predict(XCnn_test))
            customauroc=roc_auc_score(y_true=y_test,y_score=custom.predict(XCnn_test))
        elif task=="multiclass":
            customf1=f1_score(y_true=y_test,y_pred=custom.predict(XCnn_test),average="macro")
            customauroc=roc_auc_score(y_true=y_test,y_score=custom.predict_proba(XCnn_test),average="macro",multi_class="ovr")
        pd.DataFrame([[encoding,customauroc,customf1,task,"custom",dataset_name]]).to_csv(outfile,mode='a',index=False,header=False)

        resnet.fit(XCnn_train, y_train)
        if task=="binary":
            resnetf1=f1_score(y_true=y_test,y_pred=resnet.predict(XCnn_test))
            resnetauroc=roc_auc_score(y_true=y_test,y_score=resnet.predict(XCnn_test))
        elif task=="multiclass":
            resnetf1=f1_score(y_true=y_test,y_pred=resnet.predict(XCnn_test),average="macro")
            resnetauroc=roc_auc_score(y_true=y_test,y_score=resnet.predict_proba(XCnn_test),average="macro",multi_class="ovr")
        pd.DataFrame([[encoding,resnetauroc,resnetf1,task,"resnet",dataset_name]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"{encoding} encoding is done for {task} task with:\n\t-custom\n\t\t-f1: {customf1}\n\t\t-auroc: {customauroc}\n\t-resnet\n\t\t-f1: {resnetf1}\n\t\t-auroc: {resnetauroc}")
    logger.info(f"All seed variations are done for index {j}.")
    os.remove(fcgrfile)

if __name__=="__main__":
    main()