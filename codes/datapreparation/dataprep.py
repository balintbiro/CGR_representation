# import the necessary libraries
import os
import sys
import click
import random
import logging
import datetime
import requests
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent.parent
CODES=PROJECT_ROOT / "codes"
DATA=PROJECT_ROOT / "data"
RESULTS=PROJECT_ROOT / "results"

from codes.utils import loggerConfig,DeepLoc,MultiTox,PFAM,Immune

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
    "--dataset_name",
    help="Name of the dataset",
    required=True,
    type=click.Choice(
        ["deeploc","tox","pfam","immune"],
        case_sensitive=False
    )
)
@click.option(
    "--tempfile",
    help="Path to a temporary file if its not in a .csv format",
    required=True
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the clean datasets with labels",
    required=True
)

def main(
        logfile:str,
        dataset_name:str,
        tempfile:str,
        outfile:str
    )->None:
    loggerConfig(logfile=logfile)
    if dataset_name=="deeploc":
        dataset_class=DeepLoc()
    elif dataset_name=="tox":
        dataset_class=MultiTox()
    elif dataset_name=="pfam":
        dataset_class=PFAM()
    elif dataset_name=="immune":
        dataset_class=Immune()
    logger.info(f"Dataprep is running for {dataset_name}")
    status=dataset_class.get(outfile=tempfile)
    logger.info(f"Dataset was gathered with status code: {status}")
    cleaned,original_size,additional_info=dataset_class.clean(tempfile=tempfile)
    logger.info(f"The original dataset has shape: {original_size}")
    logger.info(f"The cleaned dataset has shape: {cleaned.shape}")
    if len(additional_info)>0:
        logger.info(f"Encoded labels are the followings: {additional_info}")
    os.remove(path=tempfile)
    logger.info(f"{tempfile} temporary file was removed.")
    cleaned.to_csv(outfile,index=False)
    logger.info(f"Cleaned {dataset_name} is exported to {outfile}\n")

if __name__=="__main__":
    main()