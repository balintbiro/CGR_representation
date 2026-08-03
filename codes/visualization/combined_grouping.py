from codes.utils import monomer_groupings
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

def CGRepresentation(seq:str,encodings:dict)->list:
    """
    Generating Chaos Game Representations (point coordinates) for DNA sequences.

    Parameters:
    - seq: DNA sequence with datatype of str
    - encodings: dictionary type variable that contains values for all of the nucleotides

    Returns:
    - list of coordinates
    """
    seq=seq.replace("\n",'').replace('N','')
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

prot_acc_dist=pd.read_csv("../../Downloads/random_grouping.csv")

prot_acc_dist[prot_acc_dist.accuracy==prot_acc_dist.accuracy.min()]

prot_acc_dist[prot_acc_dist.accuracy==prot_acc_dist.accuracy.max()]

def draw_edge_with_label(G,pos,edges,color,ax,label):
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges,
        edge_color=color,
        arrows=True,
        width=3,
        ax=ax
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={edges[0]: label},
        ax=ax,
        label_pos=0.5,
        font_size=5
    )

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

annots=[
    "Aliphatic",
    "Size1",
    "Size2",
    "Hydro\nphobic",
    "Positive",
    "Charged",
    "Polar",
    "Aromatic"
]
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

plt.style.use("seaborn-v0_8-colorblind")

fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(8)

ax1=plt.subplot2grid(shape=(2, 4), loc=(0, 0), colspan=1)
ax2=plt.subplot2grid(shape=(2,4), loc=(0, 1), colspan=1)

ax3=plt.subplot2grid(shape=(2,4), loc=(1, 0), colspan=1)
ax4=plt.subplot2grid(shape=(2,4), loc=(1, 1), colspan=1)

ax5=plt.subplot2grid(shape=(2,4), loc=(0,2), rowspan=2,colspan=2)
tickfontsize=5
labelfontsize=7
titlefontsize=10
#################################################################
G = nx.Graph()
G.add_edge("A", "C")
G.add_edge("A", "G")
G.add_edge("A", "T")

struc_col="#d01110"
bond_col="#0a0a00"
sigro_col="#76b947"


# Define positions manually to allow for curvature
pos = nx.spring_layout(G,seed=0)  # You can also define custom positions

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_color="#000000",ax=ax1,node_size=50)
nx.draw_networkx_labels(G, pos, font_color="#ffffff",ax=ax1,font_size=tickfontsize)

# Draw curved edge A-G using connectionstyle
draw_edge_with_label(G,pos,[("A", "C")],sigro_col,ax1,"amino")
draw_edge_with_label(G,pos,[("G", "T")],sigro_col,ax1,"keto")
draw_edge_with_label(G,pos,[("A", "G")],struc_col,ax1,"purine")
draw_edge_with_label(G,pos,[("T", "C")],struc_col,ax1,"pyrimidine")
draw_edge_with_label(G,pos,[("A", "T")],bond_col,ax1,"weak")
draw_edge_with_label(G,pos,[("G", "C")],bond_col,ax1,"strong")

legend_elements = [
    Line2D([0], [0], color=sigro_col, lw=0, marker='o', label='Side Group'),
    Line2D([0], [0], color=struc_col, lw=0, marker='o', label='Structure'),
    Line2D([0], [0], color=bond_col, lw=0, marker='o', label='H-bond')
]
ax1.legend(handles=legend_elements,bbox_to_anchor=(.45,.7),fontsize=tickfontsize,ncols=1,bbox_transform=ax1.transAxes)
ax1.set_title(label="A",x=0,y=1,fontsize=titlefontsize)

#################################################################
heatmap=sns.heatmap(
    groupings,cmap="tab10",
    linewidths=0.5,
    linecolor="grey",
    cbar=False,
    xticklabels=list(aa_order),yticklabels=annots[::-1],
    ax=ax2
)
heatmap.tick_params(axis="both",labelsize=tickfontsize)

ax2.set_xlabel(xlabel="AAs",fontdict={"fontsize":labelfontsize})
ax2.set_xticklabels(labels=ax2.get_xticklabels(),rotation=90)
ax2.set_ylabel(ylabel="AA groupings",fontdict={"fontsize":labelfontsize})
ax2.set_title(label="B",x=0,y=1,fontsize=titlefontsize)

#################################################################
dna_seqs={
    "sigro":["A","AC","ACG","ACGT"],
    "struc":["A","AG","AGC","AGCT"],
    "bond":["A","AT","ATC","ATCG"]
}

coords,firms,dash=CGRepresentation(seq="ACGT",encodings=monomer_groupings["dna"]["side_groups"])
for index,firm in enumerate(firms):
    ax3.plot(coords[index+1][0],coords[index+1][1],marker="o",color="k")
    ax3.plot(firm[0],firm[1],linestyle="-",color="k")
    ax3.plot(dash[index][0],dash[index][1],linestyle="--",color="k",alpha=.5)

markersizes=np.linspace(start=14,stop=7,num=4,dtype=int)[::-1]
for index,element in enumerate(list(monomer_groupings["dna"]["side_groups"].values())):
    ax3.plot(element[0]-0.1,element[1]+0.2,marker=f"""${dna_seqs["sigro"][index]}$""",color=sigro_col,markersize=markersizes[index])
    ax3.plot(element[0]-0.1,element[1],marker=f"""${dna_seqs["struc"][index]}$""",color=struc_col,markersize=markersizes[index])
    ax3.plot(element[0]-0.1,element[1]-0.2,marker=f"""${dna_seqs["bond"][index]}$""",color=bond_col,markersize=markersizes[index])

ax3.set(
    frame_on=False,
    xticks=[],yticks=[],
    xticklabels=[],yticklabels=[]
)
ax3.set_title(label="C",x=0,y=1,fontsize=titlefontsize)
#################################################################

sns.histplot(
    data=prot_acc_dist.rename(columns={"accuracy":"Test accuracy"}),x="Test accuracy",
    ax=ax4,kde=True,element="step",
    color="lightblue",
    bins=30
)
ax4.arrow(0.05,0.2,0,-0.1,transform=ax4.transAxes,width=0.01,color="red",label="Min: \nFGDLRMNYVTQPSEKIWCHA")
ax4.arrow(0.95,0.2,0,-0.1,transform=ax4.transAxes,width=0.01,color="green",label="Max: \nHDWNALYVFITGSCRMEKPQ")

ax4.legend(fontsize=tickfontsize)
ax4.set_xlabel(xlabel="Test accuracy",fontdict={"fontsize":labelfontsize})
ax4.set_xticklabels(labels=ax4.get_xticklabels(),rotation=90,fontdict={"fontsize":tickfontsize})
ax4.set_ylabel(ylabel="Count",fontdict={"fontsize":labelfontsize})
ax4.set_yticklabels(labels=ax4.get_yticklabels(),fontdict={"fontsize":tickfontsize})
ax4.set_title(label="D",x=0,y=1,fontsize=titlefontsize)
#################################################################

coords,firms,dash=CGRepresentation(
    seq="AAGQKDVS",
    encodings=monomer_groupings["protein"]["polar"]
)
canonical_coords=monomer_groupings["protein"]["polar"].values()
canonical_seq=''.join(monomer_groupings["protein"]["polar"].keys())
for index,element in enumerate(canonical_coords):
    if index<9:
        ax5.plot(element[0],element[1],marker=f"${canonical_seq[index]}$",color="#ff7f0e",label="lé")
    else:
        ax5.plot(element[0],element[1],marker=f"${canonical_seq[index]}$",markerfacecolor="white",markeredgecolor="black")

x0,y0=0,0
r=1.1
angles=np.linspace(start=0,stop=2*np.pi,num=20,endpoint=False)
x=x0+r*np.sin(angles)
y=y0+r*np.cos(angles)

canonical_seq=''.join(monomer_groupings["protein"]["charged"].keys())
for index,element in enumerate(x):
    if index<5:
        ax5.plot(x[index],y[index],marker=f"${canonical_seq[index]}$",color="#2ca02c",label="lé")
    else:
        ax5.plot(x[index],y[index],marker=f"${canonical_seq[index]}$",markerfacecolor="white",markeredgecolor="black")

for index,firm in enumerate(firms):
    ax5.plot(coords[index+1][0],coords[index+1][1],marker="o",color="k",markersize=4)
    ax5.plot(firm[0],firm[1],linestyle="-",color="k")
    ax5.plot(dash[index][0],dash[index][1],linestyle="--",color="k",alpha=.5)

limit=(-1.2,1.2)

growth_factor="AAGQKDVS"

ax5.set(xlim=limit,ylim=limit)
ax5.set(
    xlabel="",ylabel="",
    xticks=[],yticks=[],
    #title=f"EGF-like growth factor sequence\nCGR with 'Polar' aa grouping"
)

ax5.text(x=-0.5,y=0.3,s="Polar:",fontdict={"fontsize":labelfontsize})
for index,x_pos in enumerate(np.linspace(start=-0.2,stop=0.25,num=len(growth_factor))):
    if growth_factor[index] in sets[annots.index("Polar")]:
        ax5.text(x_pos,0.3,growth_factor[index],color="#ff7f0e")
    else:
        ax5.text(x_pos,0.3,growth_factor[index])

artificial_seq="QQLNVYEI"
ax5.text(x=-0.5,y=0.2,s="Charged:",fontdict={"fontsize":labelfontsize})
for index,x_pos in enumerate(np.linspace(start=-0.2,stop=0.25,num=len(artificial_seq))):
    if artificial_seq[index] in sets[annots.index("Charged")]:
        ax5.text(x_pos,0.2,artificial_seq[index],color="#2ca02c")
    else:
        ax5.text(x_pos,0.2,artificial_seq[index])
ax5.set_frame_on(False)
ax5.set_title(label="E",x=0,y=1,fontsize=titlefontsize)

plt.tight_layout()
plt.savefig("data/combined_encoding_figure.png",dpi=400)
