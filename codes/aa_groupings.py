import numpy as np

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
    angles=np.linspace(start=0,stop=2*np.pi,num=len(aas))
    x=x0+r*np.sin(angles)
    y=y0+r*np.cos(angles)
    
    encoded_aas={}
    for index,aa in enumerate(reordered):
        encoded_aas[aa]=[x[index],y[index]]
    all_groupings[grouping_name]=encoded_aas

all_groupings={}
aa_grouping(all_groupings=all_groupings,grouping_name="aliphatic",first_group=list("ILV"))
aa_grouping(all_groupings=all_groupings,grouping_name="size1",first_group=list("ACGS"))
aa_grouping(all_groupings=all_groupings,grouping_name="size2",first_group=list("VCTPDNAGS"))
aa_grouping(all_groupings=all_groupings,grouping_name="hydrophobic",first_group=list("PSDNREQ"))
aa_grouping(all_groupings=all_groupings,grouping_name="positive",first_group=list("HKR"))
aa_grouping(all_groupings=all_groupings,grouping_name="charged",first_group=list("HKRDE"))
aa_grouping(all_groupings=all_groupings,grouping_name="polar",first_group=list("MFILVCAGP"))
aa_grouping(all_groupings=all_groupings,grouping_name="aromatic",first_group=list("FYWH"))
all_groupings
