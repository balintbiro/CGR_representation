
import requests
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from torchvision.models import resnet18
from Bio.ExPASy import ScanProsite,Prosite
from Bio import ExPASy
import json

def tester(mtx:pd.DataFrame,dataset_name:str,skf:StratifiedKFold)->pd.DataFrame:
    """
    Performs Stratified K fold Cross Validation on a FCGR matrix.

    Parameters:
    - mtx: matrix of bonds, sigro or struc
    - dataset_name: bond, sigro or struc
    - skf: instantiated StratifiedKFold

    Returns:
    - dataframe of the specfied matric and dataset name
    """
    X,y=mtx.drop(columns=["label"]).values,mtx["label"].values
    results=[]
    for i, (train_index,test_index) in enumerate(skf.split(X,y)):
        clf=LogisticRegression(random_state=0,n_jobs=-1)
        clf.fit(X[train_index],y[train_index])
        y_pred=clf.predict(X[test_index])
        results.append(accuracy_score(y_true=y[test_index],y_pred=y_pred))
    return pd.DataFrame(
        data=[results,len(results)*[dataset_name]],
    ).T

def loggerConfig(logfile:str)->None:
    """
    Setting up the logging structure.

    Parameters:
    - logfile:str, filepath of the logfile

    Returns:
    - None
    """
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        filemode='a',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

class Cnn(nn.Module):
    def __init__(self,output_dim:int):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(10 * 16 * 16, output_dim)

    def forward(self, x):
        x = torch.relu(self.pool(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(self,output_dim:int):
        super().__init__()
        self.model=resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=3,
            bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dim)

    def forward(self, x):
        return self.model(x)

class DeepLoc:
    def __init__(self):
        self.url="https://services.healthtech.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta"

    def get(self,outfile:str)->int:
        response=requests.get(url=self.url)
        with open(outfile,"wb") as f:
            f.write(response.content)
        return response.status_code

    def clean(self,tempfile:str)->tuple:
        parser=SeqIO.parse(
            handle=tempfile,
            format="fasta"
        )
        proteinogenic_aas="ACDEFGHIKLMNPQRSTVWY"
        sequences=[]
        for record in parser:
            label=record.description.split('-')[-1]
            seq_id=record.id.split()[0][1:]
            sequences.append([seq_id,str(record.seq),label])
        # create dataframe from sequences and labels
        sequences=pd.DataFrame(data=sequences,columns=["id","sequence","label"])
        fil=sequences["sequence"].apply(lambda sequence: len(set(str(sequence))-set(proteinogenic_aas))==0)
        sequences["label"]=sequences["label"].replace(['M','S'],[0,1])
        return (sequences[sequences["label"].isin([0,1])][fil],sequences.shape,{})

class Immune:
    def init(self):
        pass

    def get(self,outfile:str)->tuple:
        splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
        train=pd.read_csv("hf://datasets/AI4Protein/VenusVaccine_VirusBinary_ESMFold/" + splits["train"])
        test=pd.read_csv("hf://datasets/AI4Protein/VenusVaccine_VirusBinary_ESMFold/" + splits["test"])
        valid=pd.read_csv("hf://datasets/AI4Protein/VenusVaccine_VirusBinary_ESMFold/" + splits["validation"])

        df=(
            pd.concat(
                [
                    train[["aa_seq","label"]],
                    test[["aa_seq","label"]],
                    valid[["aa_seq","label"]]
                ]
            )
            .reset_index(drop=True)
            .rename(columns={"aa_seq":"sequence"})
        )
        if df.shape[0]>0:
            status="success"
            df.to_csv(outfile,index=False)
        else:
            status="error"
        return status

    def clean(self,tempfile:str)->tuple:
        proteinogenic_aas="ACDEFGHIKLMNPQRSTVWY"
        sequences=pd.read_csv(tempfile)
        fil=sequences["sequence"].apply(lambda sequence: len(set(str(sequence))-set(proteinogenic_aas))==0)
        return (sequences[fil],sequences.shape,{})

class PFAM:
    def __init__(self):
        self.url="https://zenodo.org/records/8167436/files/pfam_46872x62.csv?download=1"

    def get(self,outfile:str)->int:
        response=requests.get(url=self.url)
        with open(outfile,"wb") as f:
            f.write(response.content)
        return response.status_code

    def clean(self,tempfile:str)->tuple:
        proteinogenic_aas="ACDEFGHIKLMNPQRSTVWY"
        sequences=(
            pd.read_csv(tempfile)
            .rename(columns={"family":"label"})[["sequence","label"]]
        )
        # 5 of the most prominent categories
        top5=sequences["label"].value_counts().index[:5]
        sequences=sequences[sequences["label"].isin(top5)]
        encoder=LabelEncoder()
        sequences["label"]=encoder.fit_transform(sequences["label"])
        fil=sequences["sequence"].apply(lambda sequence: len(set(str(sequence))-set(proteinogenic_aas))==0)
        label_dict=dict(zip(encoder.inverse_transform(sequences["label"].unique()),sequences["label"].unique()))
        return (sequences[fil],sequences.shape,label_dict)

class MultiTox:
    def __init__(self):
        self.url="https://raw.githubusercontent.com/cosylabiiit/MultiTox/refs/heads/main/Data/toxin3052.csv"

    def get(self,outfile:str)->int:
        response=requests.get(url=self.url)
        with open(outfile,"wb") as f:
            f.write(response.content)
        return response.status_code

    def clean(self,tempfile:str)->tuple:
        sequences=pd.read_csv(tempfile)
        sequences.rename(columns={"Sequence":"sequence","Label":"label"},inplace=True)
        proteinogenic_aas="ACDEFGHIKLMNPQRSTVWY"
        fil=sequences["sequence"].apply(lambda sequence: len(set(str(sequence))-set(proteinogenic_aas))==0)
        return (sequences[fil],sequences.shape,{})

class ProSite:
    def __init__(self,sequence:str):
        self.sequence=sequence

    def find_motives(self)->pd.DataFrame:
        scan=ScanProsite.scan(seq=self.sequence,output="json")
        data=scan.read()
        if isinstance(data,bytes):
            data=data.decode("utf-8")
        results=pd.DataFrame(json.loads(data).get("matchset"))
        return results

    def get_motives(self,row:pd.Series)->pd.DataFrame:
        start,stop,signature=row["start"],row["stop"],row["signature_ac"]
        with ExPASy.get_prosite_raw(signature) as handle:
            signature_info=Prosite.read(handle)
        return [signature,signature_info.name,signature_info.description,signature_info.pattern]