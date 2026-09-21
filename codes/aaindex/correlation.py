# import the necessary libraries
import os
import sys
import click
import random
import logging
import datetime
import requests

from codes.utils import loggerConfig

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
    "--outdir",
    help="Name of the output folder",
    required=True
)

def main(logfile:str,outdir:str)->None:
    loggerConfig(logfile=logfile)
    