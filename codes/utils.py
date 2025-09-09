side_groups={
    'A':[-1,1],
    'C':[-1,-1],
    'G':[1,-1],
    'T':[1,1]
}
structure={
    'A':[-1,1],
    'C':[1,-1],
    'G':[-1,-1],
    'T':[1,1]
}
bonds={
    'A':[-1,1],
    'C':[1,1],
    'G':[1,-1],
    'T':[-1,-1]
}

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

side_groups={
    'A':[-1,1],
    'C':[-1,-1],
    'G':[1,-1],
    'T':[1,1]
}
structure={
    'A':[-1,1],
    'C':[1,-1],
    'G':[-1,-1],
    'T':[1,1]
}
bonds={
    'A':[-1,1],
    'C':[1,1],
    'G':[1,-1],
    'T':[-1,-1]
}

def CGRepresentation(
    sequence:str,
    encodings:dict,
    scaling_factor:float
)->pd.DataFrame:
    """
    Generating Chaos Game Representations (point coordinates) for DNA sequences.

    Parameters:
    - sequence: DNA sequence with datatype of str
    - encodings: dictionary type variable that contains values for all of the nucleotides
    - scaling_factor: float for scaling the steps between nucleotides

    Returns:
    - dataframe of coordinates
    """
    sequence=sequence.replace("\n",'').replace('N','')
    coordinates=[[0,0]]

    for nucleotide in sequence:
        corner=encodings[nucleotide]
        current=coordinates[-1]
        x=scaling_factor*(corner[0]+current[0])
        y=scaling_factor*(corner[1]+current[1])
        coordinates.append([x,y])
        
    return pd.DataFrame(
        data=coordinates[1:],
        columns=list("xy")
    )

def FrequencyCGR(
        coordinates:pd.DataFrame,
        resolution:float,
        flatten=True
)->np.array:
    """
    Generate a Frequency Chaos Game Representation (FCGR) matrix.

    Parameters:
    - coordinates: dataframe of x,y coordinates representing points in the unit square [-1, 1] x [-1, 1]
    - resolution: int, the number of bins along each axis (e.g., 8 for a 8x8 matrix)
    - flatten: bool, for visualization purposes, flatten=False should be used

    Returns:
    - fcgr_matrix: 1D (2D if flatten=False is applied) numpy array of shape (resolution, resolution)
    """
    bins=np.linspace(start=-1.01,stop=1.01,num=resolution+1)
    labels=np.linspace(start=0,stop=resolution-1,num=resolution,dtype=int)
    
    categories=pd.DataFrame(columns=[list("xy")])
    categories['x']=pd.cut(x=coordinates.x,bins=bins,labels=labels)
    categories['y']=pd.cut(x=coordinates.y,bins=bins,labels=labels)

    fcgr_matrix = np.zeros((resolution, resolution), dtype=int)

    for index,row in categories.iterrows():
        fcgr_matrix[row.x,row.y] += 1
    fcgr_matrix=np.rot90(m=fcgr_matrix,axes=(-2,-1))
    if flatten:
        return fcgr_matrix.flatten()
    else:
        return fcgr_matrix

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
        clf=XGBClassifier(random_state=0)
        clf.fit(X[train_index],y[train_index])
        y_pred=clf.predict(X[test_index])
        results.append(accuracy_score(y_true=y[test_index],y_pred=y_pred))
    return pd.DataFrame(
        data=[results,len(results)*[dataset_name]],
    ).T
