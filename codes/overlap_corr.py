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
    "--n",
    help="Number of iterations for the random search",
    required=True,
    type=int
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
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the accuracies accross encodings",
    required=True
)

def main(
        logfile,
        seqfile,
        n,
        dataset_name,
        outfile
    ):
    fcgrfile,sf,res="../data/temp_encoding.csv",0.865,35
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["encoding","mean","median","std","dataset"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    proteogenic_aas="ACDEFGHIKLMNPQRSTVWY"
    for i in range(n):
        seed=i
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
        zeros=(df==0).astype(int).sum(axis=1)
        mean,median,std=(zeros/(35*35)).mean(),(zeros/(35*35)).median(),(zeros/(35*35)).std()
        pd.DataFrame([[encoding,mean,median,std,dataset_name]]).to_csv(outfile,mode='a',index=False,header=False)
        logger.info(f"{i+1} encoding is done!")


if __name__=="__main__":
    main()