# importing required libraries
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
from genomic_benchmarks.data_check import info, list_datasets
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanNontataPromoters, HumanOcrEnsembl, HumanEnhancersCohn, DrosophilaEnhancersStark

# get the predefined training and testing datasets
train=DrosophilaEnhancersStark(split="train",version=0)
test=DrosophilaEnhancersStark(split="test",version=0)

# merge the two datasets
dataset=[]
for i in train:
    dataset.append([i[0],i[1]])
    if len(dataset)%5000==0:
        print(f"{len(dataset)} sequences are processed.")

for i in test:
    dataset.append([i[0],i[1]])
    if len(dataset)%5000==0:
        print(f"{len(dataset)} sequences are processed.")

# export the sequences with labels
(
    pd.DataFrame(data=dataset,columns=["sequence","label"])
    .to_csv("../data/dros_enhancer.csv",index=False)
)

# parse the fasta file for deeploc
parser=SeqIO.parse(
    handle="../../Downloads/deeploc_1_membrane_vs_soluble.fasta",
    format="fasta"
)

sequences=[]
for record in parser:
    label=record.description.split('-')[-1]
    if (
        (label in ['S','M']) and
        ('X' not in str(record.seq)) and
        ('U' not in str(record.seq)) and
        ('B' not in str(record.seq))
    ):
        sequences.append([str(record.seq),label])

# create dataframe from sequences and labels
sequences=pd.DataFrame(data=sequences,columns=["sequence","label"])

# balancing the dataset
balanced=pd.concat(
    [
        sequences[sequences["label"]=='M'],
        sequences[sequences["label"]=="S"].sample(
            n=sequences["label"].value_counts()['M'],
            random_state=0
        )
    ]
)
balanced["label"]=balanced["label"].replace(['M','S'],[0,1])

# export the datafrae to csv file
balanced.to_csv("data/deeploc.csv",index=False)
