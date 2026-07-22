import os
import sys
import click
import random
import logging
import datetime
from aaindex import aaindex1
import numpy as np
import pandas as pd
import subprocess
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetBinaryClassifier,NeuralNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"

from codes.utils import loggerConfig,ResNet,Cnn

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def permute(sequence:str)->list:
    permuted_sequences=[]
    mismatches=[]
    for i in range(0,len(sequence)):
        subseq_const=sequence[:i]
        subseq_var=sequence[i:len(sequence)]
        for j in range(10):
            permuted_sequences.append(subseq_const+''.join(np.random.choice(a=list(subseq_var),size=len(subseq_var),replace=False)))
            mismatches.append(len(subseq_var))
    results=pd.DataFrame(columns=["original_seq","permuted_seq","mismatches"])
    results["original_seq"]=len(permuted_sequences)*[sequence]
    results["permuted_seq"]=permuted_sequences
    results["mismatches"]=mismatches
    return results

@click.command()
@click.option(
    "--logfile",
    help="Path to logfile.[log]",
    required=True
)
@click.option(
    "--seqfile",
    help="Path to seqfile.[csv]",
    required=True
)
@click.option(
    "--outfile",
    help="Path to output file.[csv] that will contain the AA indices",
    required=True
)

def main(
        logfile:str,
        seqfile:str,
        outfile:str
    )->None:
    fcgrfile,sf,res=DATA/"random_encoding_0865_35.csv",0.865,35
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["original_seq","encoding","mismatches","auroc","f1"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    # permuting aaindices ARGP820102 and 103
    # signal sequence helical potential and membrane-buried preference parameters
    sequences=pd.concat([permute(sequence="WFMLYRQNHGSIETKVDPCA"),permute(sequence="SPNQTAGRLFDEKCVWIMHY")]).reset_index(drop=True)
    for index,encoding in enumerate(sequences["permuted_seq"]):
        # set seed for reproducible results in training and testing
        seed=0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # creating the random encodings and the corresponding FCGRs
        subprocess.run(f"""Rscript --vanilla {CODES}/FCGR_gen.R --encoding {encoding} --output_file {fcgrfile} --input_filename {seqfile} --scaling_factor {sf} --resolution {res}""",shell=True)
        logger.info(f"Input matrix with {encoding} encoding is generated.")
        # getting the FCGRs and training the CNN on them
        df=pd.read_csv(fcgrfile)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        custom=NeuralNetBinaryClassifier(
            Cnn(output_dim=1),
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
            train_split=None,
            iterator_train__shuffle=None
        )
        y=df["label"].values.astype("float32")
        X=df.drop(columns=["label"]).div(df.drop(columns=["label"]).max(axis=1),axis=0).values.astype("float32")
        XCnn = X.reshape(-1, 1, res,res)
        XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=seed, stratify=y)
        custom.fit(XCnn_train, y_train)
        # get the accuracy scores on the testing data and save them in the output file
        customf1=f1_score(y_true=y_test,y_pred=custom.predict(XCnn_test))
        customauroc=roc_auc_score(y_true=y_test,y_score=custom.predict(XCnn_test))
        pd.DataFrame([[sequences["original_seq"].values[index],encoding,sequences["mismatches"].values[index],customauroc,customf1]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"{encoding} encoding is done with:\n\t-custom\n\t\t-f1: {customf1}\n\t\t-auroc: {customauroc}\n\t")
        fcgrfile.unlink()
        logger.info(f"Input matrix with {encoding} encoding is removed.\n")

if __name__=="__main__":
    main()
