import os
import sys
import click
import random
import logging
import datetime
import numpy as np
import pandas as pd
import subprocess

from utils import loggerConfig,ProSite

logger=logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

@click.command()
@click.option(
    "--logfile",
    help="Path to logfile.[log]",
    required=True
)
@click.option(
    "--seqfile",
    help="Path to seqfile.[csv]",
    required=True
)
@click.option(
    "--outdir",
    help="Path to output directory that will contain motives and signatures",
    required=True
)

def main(logfile:str,seqfile:str,outdir:str)->None:
    # create two output files if they are not already existing ones
    if os.path.exists(os.path.join(outdir,"prosite_motives.csv")):
        pass
    else:
<<<<<<< HEAD
        out_df=pd.DataFrame(columns=["sequence_id","signature_ac","start","stop"])
=======
        out_df=pd.DataFrame(columns=["sequence_ac","start","stop","signature_ac","score","level","sequence",])
>>>>>>> 5cb608f (Scanprosite code and log)
        out_df.to_csv(os.path.join(outdir,"prosite_motives.csv"),index=False)
    if os.path.exists(os.path.join(outdir,"prosite_signatures.csv")):
        pass
    else:
        out_df=pd.DataFrame(columns=["accession","name","description","pattern"])
        out_df.to_csv(os.path.join(outdir,"prosite_signatures.csv"),index=False)
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    sequences=pd.read_csv(seqfile)
<<<<<<< HEAD
    accessions=[]
    for index,sequence in enumerate(sequences["sequence"]):
        seqid=sequences["id"].values[index]
        prosite=ProSite(sequence=sequence,id=seqid)
=======
    for index,sequence in enumerate(sequences["sequence"][:5]):
        prosite=ProSite(sequence=sequence)
>>>>>>> 5cb608f (Scanprosite code and log)
        try:
            # find motives in a particular sequence
            motives=prosite.find_motives()
            if motives.shape[0]>0:
                # if theres any motives, write it out and search for signatures
                motives["sequence_id"]=seqid
                motives[["sequence_id","signature_ac","start","stop"]].to_csv(os.path.join(outdir,"prosite_motives.csv"),mode='a',index=False,header=False)
                signatures=pd.DataFrame(
                    data=motives.apply(lambda row: prosite.get_motives(row),axis=1).tolist(),
                    columns=["accession","name","description","pattern"]
                ).drop_duplicates(subset=["accession"])
                signatures[~signatures["accession"].isin(accessions)].to_csv(os.path.join(outdir,"prosite_signatures.csv"),mode='a',index=False,header=False)
                accessions+=signatures["accession"].drop_duplicates().tolist()
        except Exception as e:
            logger.info(f"Problem with {index}th sequence: {str(e)}")
        if (index!=0) & (index%100==0):
            logger.info(f"{index}th sequence (/{sequences.shape[0]}) is done scanning.")
    logger.info(f"Motives are written into {os.path.join(outdir,"prosite_motives.csv")}")
    logger.info(f"Signatures are written into {os.path.join(outdir,"prosite_signatures.csv")}")

if __name__=="__main__":
    main()
