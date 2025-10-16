import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,cross_validate

# aas classification https://www.nature.com/articles/s41598-020-72174-5
monomer_groupings={
    "dna":{
        "side_groups":{
            'A':[-1,1],
            'C':[-1,-1],
            'G':[1,-1],
            'T':[1,1]
        },
        "structure":{
            'A':[-1,1],
            'C':[1,-1],
            'G':[-1,-1],
            'T':[1,1]
        },
        "bonds":{
            'A':[-1,1],
            'C':[1,1],
            'G':[1,-1],
            'T':[-1,-1]
        }
    },
    "protein":{'aliphatic': {'I': [np.float64(0.0), np.float64(1.0)],
  'L': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'V': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'P': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'D': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'F': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'A': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'S': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'M': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'E': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'H': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'Q': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'R': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'C': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'Y': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'N': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'K': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'G': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'size1': {'A': [np.float64(0.0), np.float64(1.0)],
  'C': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'G': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'S': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'P': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'D': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'F': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'V': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'M': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'E': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'H': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'Q': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'L': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'R': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'Y': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'N': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'I': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'K': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'size2': {'V': [np.float64(0.0), np.float64(1.0)],
  'C': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'T': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'P': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'D': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'N': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'A': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'G': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'S': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'Q': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'L': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'R': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'Y': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'F': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'I': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'K': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'M': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'E': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'H': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'W': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'hydrophobic': {'P': [np.float64(0.0), np.float64(1.0)],
  'S': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'D': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'N': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'R': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'E': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'Q': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'L': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'C': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'Y': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'F': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'A': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'I': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'K': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'V': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'M': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'G': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'H': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'positive': {'H': [np.float64(0.0), np.float64(1.0)],
  'K': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'R': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'P': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'D': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'F': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'A': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'S': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'V': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'M': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'E': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'Q': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'L': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'C': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'Y': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'N': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'I': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'G': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'charged': {'H': [np.float64(0.0), np.float64(1.0)],
  'K': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'R': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'D': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'E': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'P': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'Q': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'L': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'C': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'Y': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'N': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'F': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'A': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'S': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'I': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'V': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'M': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'G': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'polar': {'M': [np.float64(0.0), np.float64(1.0)],
  'F': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'I': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'L': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'V': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'C': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'A': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'G': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'P': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'D': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'Q': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'R': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'Y': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'N': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'S': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'K': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'E': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'H': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'W': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]},
 'aromatic': {'F': [np.float64(0.0), np.float64(1.0)],
  'Y': [np.float64(0.3090169943749474), np.float64(0.9510565162951535)],
  'W': [np.float64(0.5877852522924731), np.float64(0.8090169943749475)],
  'H': [np.float64(0.8090169943749475), np.float64(0.5877852522924731)],
  'P': [np.float64(0.9510565162951535), np.float64(0.30901699437494745)],
  'D': [np.float64(1.0), np.float64(6.123233995736766e-17)],
  'A': [np.float64(0.9510565162951536), np.float64(-0.30901699437494734)],
  'S': [np.float64(0.8090169943749475), np.float64(-0.587785252292473)],
  'V': [np.float64(0.5877852522924732), np.float64(-0.8090169943749473)],
  'M': [np.float64(0.3090169943749475), np.float64(-0.9510565162951535)],
  'E': [np.float64(1.2246467991473532e-16), np.float64(-1.0)],
  'Q': [np.float64(-0.3090169943749473), np.float64(-0.9510565162951536)],
  'L': [np.float64(-0.587785252292473), np.float64(-0.8090169943749475)],
  'R': [np.float64(-0.8090169943749473), np.float64(-0.5877852522924732)],
  'C': [np.float64(-0.9510565162951535), np.float64(-0.30901699437494756)],
  'N': [np.float64(-1.0), np.float64(-1.8369701987210297e-16)],
  'I': [np.float64(-0.9510565162951536), np.float64(0.30901699437494723)],
  'K': [np.float64(-0.8090169943749476), np.float64(0.5877852522924729)],
  'G': [np.float64(-0.5877852522924734), np.float64(0.8090169943749473)],
  'T': [np.float64(-0.3090169943749477), np.float64(0.9510565162951535)]}
}}
####################################################################################
def CGRepresentation(
        sequence:str,
        encodings:dict,
        scaling_factor:float
    )->pd.DataFrame:
    """
    Generating Chaos Game Representations (point coordinates) for sequences.

    Parameters:
    - sequence: biological sequence with datatype of str
    - encodings: dictionary type variable that contains values for all of the monomers
    - scaling_factor: float for scaling the steps between nucleotides.
                      Sometimes referred to as dividing ratio. Usually 0.5 is applied for DNA

    Returns:
    - dataframe of xy coordinates
    """
    sequence=sequence.replace("\n",'').replace('N','')
    # starting point, origo
    coordinates=[[0,0]]

    # iterating through the sequence and calculating the coordinates
    for monomer in sequence:
        corner=encodings[monomer]
        current=coordinates[-1]
        x=scaling_factor*(corner[0]+current[0])
        y=scaling_factor*(corner[1]+current[1])
        coordinates.append([x,y])

    # turn the xy coordinates
    return pd.DataFrame(
        data=coordinates[1:],
        columns=list("xy")
    )

####################################################################################
def FrequencyCGR(
        coordinates:pd.DataFrame,
        resolution:float,
        flatten=True
    )->np.array:
    """
    Generates a Frequency Chaos Game Representation (FCGR) matrix.

    Parameters:
    - coordinates: dataframe of x,y coordinates representing points in a square
                   this is the output of CGRepresentation function. If the CGRepresentation was run
                   with scaling_factor<=0.5, then FCGR is in unit square
    - resolution: int, the number of bins along each axis (e.g., 8 for a 8x8 matrix)
    - flatten: bool, flatten=False should be used for visualization purposes

    Returns:
    - fcgr_matrix: 1D (2D if flatten=False is applied) numpy array of shape (resolution, resolution)
    """
    # getting categories
    all_coordinates=[coordinates.x,coordinates.y]
    start,stop=np.min(a=all_coordinates)-0.1,np.max(a=all_coordinates)+0.1
    bins=np.linspace(start=start,stop=stop,num=resolution+1)
    labels=np.linspace(start=0,stop=resolution-1,num=resolution,dtype=int)

    # mapping coordinates to categories
    categories=pd.DataFrame(columns=[list("xy")])
    categories['x']=pd.cut(x=coordinates.x,bins=bins,labels=labels)
    categories['y']=pd.cut(x=coordinates.y,bins=bins,labels=labels)

    # empty matrix in a resolution x resolution dimension
    fcgr_matrix = np.zeros((resolution, resolution), dtype=int)

    # adding values to the position at category, category
    np.add.at(fcgr_matrix,(categories.x,categories.y),1)
    fcgr_matrix=np.rot90(m=fcgr_matrix,axes=(-2,-1))

    if flatten:
        return fcgr_matrix.flatten()
    else:
        return fcgr_matrix

####################################################################################
def tester(
        mtx:pd.DataFrame,
        dataset_name:str,
        skf:StratifiedKFold,
        resolution:int,
        scaling_factor:float
    )->pd.DataFrame:
    """
    Performs Stratified K fold Cross Validation on a FCGR matrix.

    Parameters:
    - mtx: matrix of bonds, sigro or struc
    - dataset_name: bond, sigro or struc
    - skf: instantiated StratifiedKFold
    - resolution: int, the number of bins along each axis (e.g., 8 for a 8x8 matrix)
    - scaling_factor: float for scaling the steps between nucleotides.
                      Sometimes referred to as dividing ratio. Usually 0.5 is applied for DNA

    Returns:
    - dataframe of the specfied matrix, dataset name, resolution and scaling factor
    """
    clf=XGBClassifier(random_state=0)
    X,y=mtx.drop(columns=["label"]).values,mtx["label"].values
    cv_results=cross_validate(estimator=clf,X=X,y=y,cv=skf,scoring=["accuracy","roc_auc"],n_jobs=-1)
    cv_results=pd.DataFrame(cv_results)[["test_accuracy","test_roc_auc"]]
    cv_results["dataset"]=dataset_name
    cv_results["resolution"]=resolution
    cv_results["scaling_factor"]=scaling_factor
    return cv_results

####################################################################################
def aa_grouping(
        all_groupings:dict,
        grouping_name:str,
        first_group:list
    )->None:
    """
    Function to create the different aa groupings.

    Parameters:
    - all groupings: dictionary that contains all the groupings
    - grouping_name: the name of method for aa grouping (polarity, hidrophobicity etc.)
    - first_group: the list of aas that should be grouped together (for example polars)

    Returns:
    - None
    """
    aas=list("ARNDCQEGHILKMFPSTWYV")
    reordered=first_group+list(set(aas)-set(first_group))

    x0,y0=0,0
    r=1
    angles=np.linspace(start=0,stop=2*np.pi,num=len(aas),endpoint=False)
    x=x0+r*np.sin(angles)
    y=y0+r*np.cos(angles)

    encoded_aas={}
    for index,aa in enumerate(reordered):
        encoded_aas[aa]=[x[index],y[index]]
    all_groupings[grouping_name]=encoded_aas

####################################################################################
def random_grouping()->tuple:
    """
    Generating random permutations of amino acids.

    Parameters:
    - None

    Returns:
    - tuple
        - tuple[0]: new order of amino acids
        - tuple[1]: random encoding
    """
    groupings=pd.DataFrame(monomer_groupings["protein"]["polar"])
    new_order=np.random.choice(
        a=groupings.columns,
        size=len(groupings.columns),
        replace=False
    )
    groupings.columns=new_order
    random_grouping=groupings.to_dict(orient="list")
    return new_order,random_grouping
