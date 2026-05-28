# import the necessary libraries
import os
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
from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import loggerConfig,CnnBinary

logger=logging.getLogger(__name__)

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
    "--fcgrfile",
    help="Path to file[.csv] that will contain the FCGRs"
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the accuracies accross encodings",
    required=True
)
@click.option(
    "--sf",
    help="Scaling factor for CGR",
    required=True,
    type=float
)
@click.option(
    "--res",
    help="Resolution (int) of the FCGR",
    required=True,
    type=int
)
@click.option(
    "--n",
    help="Number of iterations for the random search",
    required=True,
    type=int
)

def main(
        logfile,
        seqfile,
        fcgrfile,
        outfile,
        sf,
        res,
        n
    ):
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["encoding","accuracy"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    proteogenic_aas="ACDEFGHIKLMNPQRSTVWY"
    for i in range(n):
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
        X,y=df.drop(columns=["label"]).div(df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32"),df["label"].values.astype("float32")
        XCnn = X.reshape(-1, 1, res,res)
        XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=seed, stratify=y)
        cnn = NeuralNetBinaryClassifier(
            CnnBinary,
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
            train_split=None,
            iterator_train__shuffle=False
        )
        cnn.fit(XCnn_train, y_train)
        y_pred=cnn.predict(XCnn_test)
        # get the accuracy scores on the testing data and save them in the output file
        acc=accuracy_score(y_true=y_test,y_pred=y_pred)
        pd.DataFrame([[encoding,acc]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"Accuracy is {acc} with {encoding} encoding.")
        os.remove(fcgrfile)

if __name__=="__main__":
    main()