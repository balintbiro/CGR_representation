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
from sklearn.model_selection import StratifiedKFold,train_test_split

from utils import loggerConfig,Cnn

device="cuda" if torch.cuda.is_available() else "cpu"

logger=logging.getLogger(__name__)

def augment(
    original_data:pd.DataFrame,
    to_add:pd.DataFrame,
    original_ratio:float,
    )->pd.DataFrame:
    no_rows=int((1/(original_ratio/(1-original_ratio)))*original_data.shape[0])
    sample=to_add.sample(n=no_rows,random_state=0)
    augmented=pd.concat([
        original_data,
        sample
    ])
    return augmented

def cross_validate(
        X:pd.DataFrame|np.ndarray,
        y:pd.Series|np.ndarray,
        test:pd.DataFrame|np.ndarray,
        dataset:str,
        original_ratio:float,
        n_split=3
    ):
    skf=StratifiedKFold(n_splits=n_split,random_state=0,shuffle=True)
    acc_scores=[]
    for i,(train_index,test_index) in enumerate(skf.split(X,y)):
        X_train,X_test,y_train,y_test=X[train_index],X[test_index],y[train_index],y[test_index]
        torch.manual_seed(0)
        cnn=NeuralNetBinaryClassifier(
            Cnn,
            max_epochs=10,
            optimizer=torch.optim.Adam,
            device=device
        )
        cnn.fit(X_train,y_train)
        Cnntest=test.drop(columns=["label","dataset"]).div(test.drop(columns=["label","dataset"]).max(axis=1),axis=0).values.astype("float32")
        y_pred=cnn.predict(Cnntest.reshape(-1,1,35,35))
        y_test=test["label"].values.astype("float32")
        acc_score=accuracy_score(y_true=y_test,y_pred=y_pred)
        acc_scores.append(acc_score)
    results=pd.DataFrame()
    results["Accuracy"]=acc_scores
    results["Dataset"]=dataset.capitalize()
    results["Ratio"]=original_ratio
    return results


@click.command()
@click.option(
    "--logfile",
    help="Path to logfile[.log]",
    required=True
)
@click.option(
    "--datasets_dir",
    help="Path to directory containing dataset files[.csv]",
    required=True
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the accuracy scores regarding to datasets",
    required=True
)
@click.option(
    "--mix",
    help="Whether to mix the datasets for augmentation or not",
    type=bool,
    default=False
)

def main(
        logfile:str,
        datasets_dir:str,
        outfile:str,
        mix:bool
    )->None:
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["Accuracy","Dataset","Ratio"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    logger.info(f"Starting augmentation process with mix={mix}")
    datasets=[pd.read_csv(os.path.join(datasets_dir, file)) for file in os.listdir(datasets_dir)]
    dataset_names=[file.split(".csv")[0] for file in os.listdir(datasets_dir)]
    train,test=train_test_split(datasets[0],test_size=0.25,random_state=0)
    training_datasets=[]
    test_dataset=""
    for dataset, name in zip(datasets, dataset_names):
        dataset["dataset"]=name
        training_datasets.append(dataset.iloc[train.index])
        if name=="min":
            test_dataset=dataset
    training_datasets=pd.concat(training_datasets,axis=0)
    to_add=training_datasets[training_datasets["dataset"]!="min"]
    if mix:
        ratios=[.2,.3,.4,.5,.6,.7,.8,.9]
        for ratio in ratios:
            data=augment(
                original_data=training_datasets[training_datasets["dataset"]=="min"],
                to_add=to_add,
                original_ratio=ratio
            ).reset_index(drop=True)
            X=data.drop(columns=["label","dataset"]).div(data.drop(columns=["label","dataset"]).max(axis=1),axis=0).values.astype("float32")
            y=data["label"].values.astype("float32")
            XCnn=X.reshape(-1,1,35,35)
            accuracy_scores=cross_validate(
                X=XCnn,y=y,
                test=test_dataset,
                dataset="mix",
                original_ratio=ratio,
                n_split=10
            )
            accuracy_scores.to_csv(outfile,mode='a',index=False,header=False)
            logger.info(f"Augmentation CV with:\n\t-mix={mix}\n\t-original dataset ratio={ratio}")
    else:
        ratios=[.5,.6,.7,.8,.9]
        for dataset in to_add["dataset"].unique():
            selection=to_add[to_add["dataset"]==dataset]
            for ratio in ratios:
                data=augment(
                    original_data=training_datasets[training_datasets["dataset"]=="min"],
                    to_add=selection,
                    original_ratio=ratio
                ).reset_index(drop=True)
                X=data.drop(columns=["label","dataset"]).div(data.drop(columns=["label","dataset"]).max(axis=1),axis=0).values.astype("float32")
                y=data["label"].values.astype("float32")
                XCnn=X.reshape(-1,1,35,35)
                accuracy_scores=cross_validate(
                    X=XCnn,y=y,
                    test=test_dataset,
                    dataset=dataset,
                    original_ratio=ratio,
                    n_split=10
                )
                accuracy_scores.to_csv(outfile,mode='a',index=False,header=False)
                logger.info(f"Augmentation CV with:\n\t-mix={mix}\n\t-dataset={dataset}\n\t-original dataset ratio={ratio}")


if __name__=="__main__":
    main()