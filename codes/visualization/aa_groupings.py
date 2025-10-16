import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,roc_auc_score

from utils import monomer_groupings
import seaborn as sns

import matplotlib.colors as mcolors

# Get the 'tab10' colormap
tab10 = plt.get_cmap('tab10')

# Generate hexadecimal color codes
hex_codes = [mcolors.to_hex(tab10(i)) for i in range(tab10.N)]
hex_codes[:8]

sets = [
        set("ILV"),
        set("ACGS"),
        set("VCTPDNAGS"),
        set("PSDNREQ"),
        set("HKR"),
        set("HKRDE"),
        set("MFILVCAGP"),
        set("FYWH")
    ]
annots=[
    "Aliphatic",
    "Size1",
    "Size2",
    "Hydrophobic",
    "Positive",
    "Charged",
    "Polar",
    "Aromatic"
]

def CGRepresentation(seq:str,encodings:dict)->list:
    """
    Generating Chaos Game Representations (point coordinates) for DNA sequences.

    Parameters:
    - seq: DNA sequence with datatype of str
    - encodings: dictionary type variable that contains values for all of the nucleotides

    Returns:
    - list of coordinates
    """
    seq=seq.replace("\n",'')
    coordinates=[[0,0]]
    firm,dash=[],[]
    for nuc in seq:
        corner=encodings[nuc]
        current=coordinates[-1]
        x=(corner[0]+current[0])/2
        y=(corner[1]+current[1])/2
        coordinates.append([x,y])
        firm.append([[current[0],x],[current[1],y]])
        dash.append([[x,corner[0]],[y,corner[1]]])
    return coordinates,firm,dash

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))

aa_order="RKHWYFMILVTSACGPNDEQ"
groupings=[
    #"RKHWYFMILVTSACGPNDEQ"
    [np.nan,np.nan,1,1,1,1,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],# aromatic
    [np.nan,np.nan,np.nan,np.nan,np.nan,2,2,2,2,2,np.nan,np.nan,2,2,2,2,np.nan,np.nan,np.nan,np.nan],# polar
    [3,3,3,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,3,3,np.nan],# charged
    [4,4,4,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],# positive
    [5,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,5,np.nan,np.nan,np.nan,5,5,5,5,5],# hydrophobic
    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,6,6,6,6,6,6,6,6,6,np.nan,np.nan],# size2
    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,7,7,7,7,np.nan,np.nan,np.nan,np.nan,np.nan],# size1
    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,8,8,8,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]# aliphatic
]
sns.heatmap(
    groupings,cmap="tab10",
    linewidths=0.5,
    linecolor="grey",
    cbar=False,
    xticklabels=list(aa_order),yticklabels=annots[::-1],
    ax=ax1
)
ax1.set(xlabel="Amino acids",ylabel="Amino acid groupings")

coords,firms,dash=CGRepresentation(
    seq="AAGQKDVS",
    encodings=monomer_groupings["protein"]["polar"]
)
canonical_coords=monomer_groupings["protein"]["polar"].values()
canonical_seq=''.join(monomer_groupings["protein"]["polar"].keys())
for index,element in enumerate(canonical_coords):
    if index<9:
        ax2.plot(element[0],element[1],marker=f"${canonical_seq[index]}$",color="#ff7f0e",label="lé")
    else:
        ax2.plot(element[0],element[1],marker=f"${canonical_seq[index]}$",markerfacecolor="white",markeredgecolor="black")

x0,y0=0,0
r=1.1
angles=np.linspace(start=0,stop=2*np.pi,num=20,endpoint=False)
x=x0+r*np.sin(angles)
y=y0+r*np.cos(angles)

canonical_seq=''.join(monomer_groupings["protein"]["charged"].keys())
for index,element in enumerate(x):
    if index<5:
        ax2.plot(x[index],y[index],marker=f"${canonical_seq[index]}$",color="#2ca02c",label="lé")
    else:
        ax2.plot(x[index],y[index],marker=f"${canonical_seq[index]}$",markerfacecolor="white",markeredgecolor="black")

for index,firm in enumerate(firms):
    ax2.plot(coords[index+1][0],coords[index+1][1],marker="o",color="k")
    ax2.plot(firm[0],firm[1],linestyle="-",color="k")
    ax2.plot(dash[index][0],dash[index][1],linestyle="--",color="k",alpha=.5)

limit=(-1.2,1.2)

growth_factor="AAGQKDVS"

ax2.set(xlim=limit,ylim=limit)
ax2.set(
    xlabel="",ylabel="",
    xticks=[],yticks=[],
    #title=f"EGF-like growth factor sequence\nCGR with 'Polar' aa grouping"
)

ax2.text(x=-1,y=-1.3,s="Polar:")
for index,x_pos in enumerate(np.linspace(start=0,stop=1,num=len(growth_factor))):
    if growth_factor[index] in sets[annots.index("Polar")]:
        ax2.text(x_pos,-1.3,growth_factor[index],color="#ff7f0e")
    else:
        ax2.text(x_pos,-1.3,growth_factor[index])

artificial_seq="QQLNVYEI"
ax2.text(x=-1,y=-1.4,s="Charged:")
for index,x_pos in enumerate(np.linspace(start=0,stop=1,num=len(artificial_seq))):
    if artificial_seq[index] in sets[annots.index("Charged")]:
        ax2.text(x_pos,-1.4,artificial_seq[index],color="#2ca02c")
    else:
        ax2.text(x_pos,-1.4,artificial_seq[index])





plt.tight_layout()
plt.savefig("data/aa_groupings.png",dpi=400)
