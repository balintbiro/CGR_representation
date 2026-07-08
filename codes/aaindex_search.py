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

from utils import loggerConfig,PreProcess,ResNet,Cnn

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def permute(sequence:str)->list:
    permuted_sequences=[]
    for i in range(0,len(sequence)):
        subseq_const=sequence[:i]
        subseq_var=sequence[i:len(sequence)]
        for j in range(3):
            permuted_sequences.append(subseq_const+''.join(np.random.chocie(a=list(subseq_var),size=len(subseq_var),replace=False)))
    return permuted_sequences

@click.command()
@click.option(
    "--logfile",
    help="Path to logfile.[log]",
    required=True
)
@click.option(
    "--outfile",
    help="Path to output file.[csv] that will contain the AA indices",
    required=True
)

def main(
        logfile:str,
        outfile:str
    )->None:
    loggerConfig(logfile=logfile)
    pp=PreProcess(aaindex1)
    n_all,all_idx=pp.check_all()
    logger.info(f"There are {n_all} available AA indices.")
    n_ioi,iois=pp.check_ioi(all_indices=all_idx)
    logger.info(f"There are {n_ioi} AA indices that have distinct values for all the proteinogenic AAs.")
    values=pp.collect_values(ioi=iois)
    scaled=pp.minmax_scaling(values=values)
    scaled.to_csv(outfile)
    logger.info(scaled.apply(lambda row: "".join(row.sort_values().index),axis=1))
    logger.info(f"ioi AA indices are scaled and written into {outfile}")
    logger.info(f"The resulting table has\n\t- physicochemical properties as indices\n\t- AAs as columns")
    logger.info(f"3x3 sample: \n{scaled.sample(n=3,random_state=0)[scaled.columns[:3]]}")

if __name__=="__main__":
    main()
