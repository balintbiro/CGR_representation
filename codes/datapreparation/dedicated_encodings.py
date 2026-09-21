import os
import sys
import click
import random
import logging
import datetime
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"

from codes.utils import loggerConfig

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

@click.command()
@click.option(
    "--logfile",
    help="Path to logfile.[log]",
    required=True
)
@click.option(
    "--model",
    help="Model name",
    required=True,
    type=click.Choice(
        ["custom","resnet"],
        case_sensitive=False
    )
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
    "--outdir",
    help="Path to output directory that will contain the dedicated encodings",
    required=True
)

def main(
        logfile:str,
        model:str,
        dataset_name:str,
        outdir:str
    )->None:
    """
    Generates dedicated encodings (min,01,02,03 and max) for the specified:

    - dataset
    - model
    - output folder
    """
    sf,res=0.865,35
    Path.mkdir(Path(outdir,dataset_name,model),parents=True,exist_ok=True)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    results=pd.read_csv(RESULTS/f"{dataset_name}_search.csv")
    results=results[results["model"]==model].sort_values(by="f1").reset_index(drop=True).iloc[[0,249,499,749,999]]
    sequences=results["encoding"]
    names=["min","q1","q2","q3","max"]
    for index,encoding in enumerate(sequences):
        # creating the encodings and the corresponding FCGRs
        subprocess.run(f"""Rscript --vanilla {CODES}/FCGR_gen.R --encoding {encoding} --output_file {Path(outdir,dataset_name,model)}/{names[index]}.csv --input_filename {DATA}/{dataset_name}_clean.csv --scaling_factor {sf} --resolution {res}""",shell=True)
        logger.info(f"Input matrix with {encoding} / {names[index]} encoding is generated.")
    logger.info(f"All encodings donewith the following parameters:\n\t-logfile: {logfile}\n\t-model: {model}\n\t-dataset name: {dataset_name}\n\t-outdir: {outdir}\n\n")

if __name__=="__main__":
    main()
