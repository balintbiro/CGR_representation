import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    "protein":{
        'aliphatic': {
            'I': [np.float64(0.0), np.float64(1.0)],
            'L': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'V': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'E': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'Q': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'C': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'N': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'H': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'Y': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'A': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'T': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'S': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'P': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'R': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'F': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'W': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'K': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'M': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'G': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'D': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'size1': {
            'A': [np.float64(0.0), np.float64(1.0)],
            'C': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'G': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'S': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'V': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'L': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'E': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'Q': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'N': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'H': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'Y': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'T': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'P': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'R': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'F': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'W': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'K': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'M': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'I': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'D': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'size2': {
            'V': [np.float64(0.0), np.float64(1.0)],
            'C': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'T': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'P': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'D': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'N': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'A': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'G': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'S': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'R': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'F': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'W': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'K': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'L': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'E': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'Q': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'M': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'Y': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'H': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'I': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'hydrophobic': {
            'P': [np.float64(0.0), np.float64(1.0)],
            'S': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'D': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'N': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'R': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'E': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'Q': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'W': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'F': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'K': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'G': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'V': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'L': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'C': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'M': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'H': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'Y': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'I': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'A': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'T': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'positive': {
            'H': [np.float64(0.0), np.float64(1.0)],
            'K': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'R': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'V': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'L': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'E': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'Q': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'C': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'N': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'Y': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'A': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'T': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'S': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'P': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'F': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'W': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'M': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'I': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'G': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'D': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'charged': {
            'H': [np.float64(0.0), np.float64(1.0)],
            'K': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'R': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'D': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'E': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'P': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'W': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'S': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'F': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'G': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'V': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'L': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'Q': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'C': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'M': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'N': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'Y': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'I': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'A': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'T': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
        'polar': {
            'M': [np.float64(0.0), np.float64(1.0)],
            'F': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'I': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'L': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'V': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'C': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'A': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'G': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'P': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'R': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'S': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'W': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'K': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'E': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'Q': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'N': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'Y': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'H': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'D': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'T': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
        },
         'aromatic': {
             'F': [np.float64(0.0), np.float64(1.0)],
            'Y': [np.float64(0.32469946920468346), np.float64(0.9458172417006346)],
            'W': [np.float64(0.6142127126896678), np.float64(0.7891405093963936)],
            'H': [np.float64(0.8371664782625285), np.float64(0.5469481581224269)],
            'V': [np.float64(0.9694002659393304), np.float64(0.24548548714079924)],
            'L': [np.float64(0.9965844930066698), np.float64(-0.08257934547233227)],
            'E': [np.float64(0.9157733266550575), np.float64(-0.40169542465296926)],
            'Q': [np.float64(0.7357239106731317), np.float64(-0.6772815716257409)],
            'C': [np.float64(0.4759473930370737), np.float64(-0.879473751206489)],
            'N': [np.float64(0.16459459028073403), np.float64(-0.9863613034027223)],
            'A': [np.float64(-0.16459459028073378), np.float64(-0.9863613034027224)],
            'T': [np.float64(-0.47594739303707345), np.float64(-0.8794737512064891)],
            'S': [np.float64(-0.7357239106731313), np.float64(-0.6772815716257414)],
            'P': [np.float64(-0.9157733266550573), np.float64(-0.40169542465296987)],
            'R': [np.float64(-0.9965844930066698), np.float64(-0.08257934547233274)],
            'K': [np.float64(-0.9694002659393305), np.float64(0.2454854871407988)],
            'M': [np.float64(-0.8371664782625288), np.float64(0.5469481581224266)],
            'I': [np.float64(-0.614212712689668), np.float64(0.7891405093963934)],
            'G': [np.float64(-0.32469946920468373), np.float64(0.9458172417006346)],
            'D': [np.float64(-2.4492935982947064e-16), np.float64(1.0)]
         }
    }
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

    np.add.at(fcgr_matrix,(categories.x,categories.y),1)
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
