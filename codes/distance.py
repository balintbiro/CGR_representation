import click
import logging
import datetime
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from utils import loggerConfig,Cnn

logger=logging.getLogger(__name__)

@click.command()
@click.option(
    "--logfile",
    help="Path to logfile[.log]",
    required=True
)
@click.option(
    "--fcgr_matrix",
    help="Path to file[.csv] containing FCGRs",
    required=True
)
@click.option(
    "--outfile",
    help="Path to outfile[.csv] that will contain the distances",
    required=True
)
@click.option(
    "--res",
    help="Resolution (int) of the FCGR",
    required=True,
    type=int
)
@click.option(
    "--n",
    help="Number of items for distance calculation",
    required=True,
    type=int
)

def get_distance(imgs:list)->np.ndarray:
    """
    Calculate the pairwise structural similarity index (SSIM) between images.

    Parameters:
    - imgs: List of 2D numpy arrays representing images.

    Returns:
    - A 2D numpy array containing the pairwise SSIM distances.
    """
    num_imgs=len(imgs)
    distance_matrix=np.zeros((num_imgs,num_imgs))

    for index1,img1 in enumerate(imgs):
        for index2,img2 in enumerate(imgs):
            distance_matrix[index1,index2]=ssim(img1,img2,data_range=1)
    return distance_matrix

def main(
        logfile,
        fcgr_matrix,
        outfile,
        res,
        n
    ):
    loggerConfig(logfile=logfile)
    script_name=os.path.basename(__file__)
    logger.info(f"Filename: {script_name} started.")
    cgr_df=pd.read_csv(cgr_matrix)
    fcgr_matrix=pd.concat(
        [
            fcgr_matrix[fcgr_matrix["label"]==0].sample(n=n,random_state=0),
            fcgr_matrix[fcgr_matrix["label"]==1].sample(n=n,random_state=0)
        ]
    )
    imgs=fcgr_matrix.drop(columns=["label"]).values.reshape((res,res))
    distance_matrix=get_distance(imgs=imgs)
    distance_df=pd.DataFrame(data=distance_matrix)
    distance_df.to_csv(outfile,index=False)
    logger.info(f"Filename: {script_name} finished.")
    
if __name__=="__main__":
    main()