# import necessary libraries
import os
import sys
import click
import torch
import random
import logging
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from skorch import NeuralNetBinaryClassifier,NeuralNetClassifier

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score

from codes.utils import loggerConfig,Cnn,ResNet

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"

device="cuda" if torch.cuda.is_available() else "cpu"

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def augment(
        original_data:pd.DataFrame,
        to_add:pd.DataFrame,
        original_ratio:float
    )->pd.DataFrame:
    """
    Augment the original data with additional samples based on the specified ratio.
    Parameters:
    - original_data: DataFrame containing the original dataset.
    - to_add: DataFrame containing the additional samples to be added for augmentation.
    - original_ratio: The desired ratio of the original dataset in the augmented dataset.
        For example, if original_ratio is 0.7, the augmented dataset will contain 70% of samples from the original dataset and 30% from the additional samples.

    Returns:
    - Augmented DataFrame containing the combined samples from the original dataset and the additional samples based on the specified ratio.
    """
    augmented_size=int(original_data.shape[0]/original_ratio)
    sample_size=int(augmented_size-original_data.shape[0])
    augmented=pd.concat(
        [
            original_data,
            to_add.sample(n=sample_size,random_state=0)
        ]
    )
    return augmented

def cross_validate(
        X:pd.DataFrame|np.ndarray,
        y:pd.Series|np.ndarray,
        test:pd.DataFrame|np.ndarray,
        dataset:str,
        original_ratio:float,
        model:str,
        task:str,
        n_split=3
    )->pd.DataFrame:
    """
    Perform cross-validation on the given dataset with the specified augmentation ratio.
    Parameters:
    - X: Feature matrix.
    - y: Label vector.
    - test: Test dataset.
    - dataset: Name of the dataset.
    - original_ratio: The desired ratio of the original dataset in the augmented dataset.
    - task: task to perform binary|multiclass
    - model: model to use, custom|resnet
    - n_split: Number of splits for cross-validation.

    Returns:
    - DataFrame containing the cross-validation results.
    """
    # skf object
    skf=StratifiedKFold(n_splits=n_split,random_state=0,shuffle=True)
    if task=="binary":
        y_test=test["label"].values.astype("float32")
        y=y.values.astype("float32")
    elif task=="multiclass":
        y_test=test["label"].values.astype(np.int64)
        y=y.values.astype(np.int64)
    # container variables for overall and class-wise accuracies
    scores=[]
    for i,(train_index,test_index) in enumerate(skf.split(X,y)):
        X_train,X_test,y_train,y_val=X[train_index],X[test_index],y[train_index],y[test_index]
        seed=0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if task=="binary":
            if model=="custom":
                cnn=NeuralNetBinaryClassifier(
                    Cnn(output_dim=1),
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
            elif model=="resnet":
                cnn=NeuralNetBinaryClassifier(
                    ResNet(output_dim=1),
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
        elif task=="multiclass":
            if model=="custom":
                cnn=NeuralNetClassifier(
                    Cnn(output_dim=5),
                    criterion=torch.nn.CrossEntropyLoss,
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
            elif model=="resnet":
                cnn=NeuralNetClassifier(
                    ResNet(output_dim=5),
                    criterion=torch.nn.CrossEntropyLoss,
                    max_epochs=10,
                    lr=0.001,
                    optimizer=torch.optim.Adam,
                    device=device,
                    train_split=None,
                    iterator_train__shuffle=None
                )
        cnn.fit(X_train,y_train)
        Cnntest=test.drop(columns=["label","dataset"]).div(test.drop(columns=["label","dataset"]).max(axis=1),axis=0).values.astype("float32")
        y_pred=pd.Series(cnn.predict(Cnntest.reshape(-1,1,35,35)))
        if task=="binary":
            score=f1_score(y_true=y_test,y_pred=y_pred)
        elif task=="multiclass":
            score=f1_score(y_true=y_test,y_pred=y_pred,average="macro")
        scores.append(score)
    results=pd.DataFrame()
    results["f1"]=scores
    results["dataset"]=dataset.capitalize()
    return results


@click.command()
@click.option(
    "--logfile",
    help="Path to logfile[.log]",
    required=True
)
@click.option(
    "--data_dir",
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
@click.option(
    "--name",
    help="Name of the dataset that is used for CV.",
    required=True,
    type=click.Choice(
            ["deeploc","immune","pfam"],
            case_sensitive=False
        )
)
@click.option(
    "--model",
    help="Name of the model to use.",
    required=True,
    type=click.Choice(
        ["custom","resnet"],
        case_sensitive=False
    )
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
    "--n",
    help="Number of cross validation splits",
    required=True,
    type=int
)

def main(
        logfile:str,
        data_dir:str,
        outfile:str,
        mix:bool,
        name:str,
        model:str,
        task:str,
        n:int
    )->None:
    """
    Performs n fold, stratified cross validation using the specified dataset, model and strategy. Strategy can be mix or single.
    """
    if os.path.exists(outfile):
        pass
    else:
        out_df=pd.DataFrame(columns=["f1","dataset","model","odr","rank"])
        out_df.to_csv(outfile,index=False)
    loggerConfig(logfile=logfile)
    logger.info(f"Starting augmentation process with mix={mix}")
    datasets=[pd.read_csv(os.path.join(data_dir, file)) for file in os.listdir(data_dir)]
    dataset_names=[file.split(".csv")[0] for file in os.listdir(data_dir)]
    train,test=train_test_split(datasets[0],test_size=0.25,random_state=0,stratify=datasets[0]["label"])
    training_datasets=[]
    test_dataset=""
    for dataset, dataset_name in zip(datasets, dataset_names):
        dataset["dataset"]=dataset_name
        training_datasets.append(dataset.iloc[train.index])
        if dataset_name=="min":
            test_dataset=dataset.iloc[test.index]
    training_datasets=pd.concat(training_datasets,axis=0)
    to_add=training_datasets[training_datasets["dataset"]!="min"]
    if mix:
        ratios=[.3,.4,.5,.6,.7,.8,.9]
        for ratio in ratios:
            data=augment(
                original_data=training_datasets[training_datasets["dataset"]=="min"],
                to_add=to_add,
                original_ratio=ratio
            ).reset_index(drop=True)
            X=data.drop(columns=["label","dataset"]).div(data.drop(columns=["label","dataset"]).max(axis=1),axis=0).values.astype("float32")
            y=data["label"]
            XCnn=X.reshape(-1,1,35,35)
            scores=cross_validate(
                X=XCnn,y=y,
                test=test_dataset,
                dataset=name,
                original_ratio=ratio,
                model=model,
                task=task,
                n_split=n
            )
            scores["model"]=model
            scores["odr"]=ratio
            scores["rank"]="mix"
            scores.to_csv(outfile,mode='a',index=False,header=False)
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
                y=data["label"]
                XCnn=X.reshape(-1,1,35,35)
                scores=cross_validate(
                    X=XCnn,y=y,
                    test=test_dataset,
                    dataset=name,
                    original_ratio=ratio,
                    model=model,
                    task=task,
                    n_split=n
                )
                scores["model"]=model
                scores["odr"]=ratio
                scores["rank"]=dataset
                scores.to_csv(outfile,mode='a',index=False,header=False)
                logger.info(f"Augmentation CV with:\n\t-mix={mix}\n\t-dataset={dataset}\n\t-original dataset ratio={ratio}")


if __name__=="__main__":
    main()