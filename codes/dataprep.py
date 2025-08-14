# importing required libraries
import numpy as np
import pandas as pd
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
