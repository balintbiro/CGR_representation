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
from sklearn.metrics import roc_auc_score,accuracy_score

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
    "--n-encoding",
    help="Number of iterations for the random search",
    required=True,
    type=int
)
@click.option(
    "--n-seed",
    help="Number of seed variations",
    required=True,
    type=int
)

def main(
        logfile,
        seqfile,
        outfile,
        n_encoding,
        n_seed
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
    for i in range(n_encoding):
        # set "random" (changing in every iterations) seed for encoding generation
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)
        encoding=''.join(np.random.choice(a=list(proteogenic_aas),size=len(proteogenic_aas),replace=False))

        # set seed for reproducible results in training and testing
        seed=0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # creating the random encodings and the corresponding FCGRs
        subprocess.run(f"""Rscript --vanilla FCGR_gen.R --encoding {encoding} --output_file {fcgrfile} --input_filename {seqfile} --scaling_factor {sf} --resolution {res}""",shell=True)
        # getting the FCGRs and training the CNN on them
        df=pd.read_csv(fcgrfile)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y=df["label"].values.astype("float32")
        X=df.drop(columns=["label"]).div(df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32")
        XCnn = X.reshape(-1, 1, res,res)
        XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=seed, stratify=y)
        model=Cnn(output_dim=1)
        cnn=NeuralNetBinaryClassifier(
            model
        )
        cnn.set_params(
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
            train_split=None,
            iterator_train__shuffle=None
        )
        cnn.fit(XCnn_train, y_train)
        acc=accuracy_score(y_true=y_test,y_pred=cnn.predict(XCnn_test))
        auroc=roc_auc_score(y_true=y_test,y_score=cnn.predict(XCnn_test))
        pd.DataFrame([[encoding,auroc,acc,"fix"]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"Accuracy is {acc} and auroc is {auroc} with {encoding} encoding and index {i}.")
        for j in range(n_seed):
            torch.manual_seed(j)
            np.random.seed(j)
            random.seed(j)
        
            model=Cnn(output_dim=1)
            cnn=NeuralNetBinaryClassifier(
                model
            )
            y=df["label"].values.astype("float32")
            cnn.set_params(
                max_epochs=10,
                lr=0.001,
                optimizer=torch.optim.Adam,
                device=device,
                train_split=None,
                iterator_train__shuffle=None
            )
            X=df.drop(columns=["label"]).div(df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32")
            XCnn = X.reshape(-1, 1, res,res)
            XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=seed, stratify=y)
            cnn.fit(XCnn_train, y_train)
            # get the accuracy scores on the testing data and save them in the output file
            acc=accuracy_score(y_true=y_test,y_pred=cnn.predict(XCnn_test))
            auroc=roc_auc_score(y_true=y_test,y_score=cnn.predict(XCnn_test))
            pd.DataFrame([[encoding,auroc,acc,"var"]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"All seed variations are done for index {i}.")
        os.remove(fcgrfile)

if __name__=="__main__":
    main()