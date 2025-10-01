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

fig,axs=plt.subplots(1,2)
G = nx.Graph()
G.add_edge("A", "C")
G.add_edge("A", "G")
G.add_edge("A", "T")

# Define positions manually to allow for curvature
pos = nx.spring_layout(G)  # You can also define custom positions

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_color="#000000",ax=axs[0])
nx.draw_networkx_labels(G, pos, font_color="#ffffff",ax=axs[0])

kwargs={"linestyle":"--"}

connection_width=6
# Draw curved edge A-G using connectionstyle
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "C")],
    edge_color="#bd8e83",
    arrows=True,
    width=connection_width,
    ax=axs[0]
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("G", "T")],
    edge_color="#bd8e83",
    arrows=True,
    style="--",
    width=connection_width,
    ax=axs[0]
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "G")],
    edge_color="#e9dac4",
    arrows=True,
    width=connection_width,
    ax=axs[0]
    #connectionstyle="arc3,rad=.3"  # Adjust rad for curvature
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("C", "T")],
    edge_color="#e9dac4",
    arrows=True,
    style="--",
    width=connection_width,
    label="Strong",
    ax=axs[0]
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "T")],
    edge_color="#749eb2",
    arrows=True,
    width=connection_width,
    ax=axs[0]
    #connectionstyle="arc3,rad=.3"  # Adjust rad for curvature
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("C", "G")],
    edge_color="#749eb2",
    arrows=True,
    style="--",
    width=connection_width,
    ax=axs[0]
)


legend_elements = [
    Line2D([0], [0], color='#bd8e83', lw=0, marker='o', label='Side Group'),
    Line2D([0], [0], color='#e9dac4', lw=0, marker='o', label='Structure'),
    Line2D([0], [0], color='#749eb2', lw=0, marker='o', label='H-bond')
]
axs[0].legend(handles=legend_elements,bbox_to_anchor=(0.5,-.1))

for spine in axs[0].spines.values():
    spine.set_visible(False)


coords,firms,dash=CGRepresentation(seq="ACGT",encodings=side_groups)
for index,firm in enumerate(firms):
    axs[1].plot(coords[index+1][0],coords[index+1][1],marker="o",color="k")
    axs[1].plot(firm[0],firm[1],linestyle="-",color="k")
    axs[1].plot(dash[index][0],dash[index][1],linestyle="--",color="k",alpha=.5)

fontsize=20

axs[1].text(x=-1.25,y=1,s='A',fontsize=fontsize,color="#749eb2",weight="semibold")
axs[1].text(x=-1.25,y=1.25,s='A',fontsize=fontsize,color="#e9dac4",weight="semibold")
axs[1].text(x=-1.25,y=1.5,s='A',fontsize=fontsize,color="#bd8e83",weight="semibold")

axs[1].text(x=-1.4,y=-1.25,s='AC',fontsize=fontsize,color="#bd8e83",weight="semibold")
axs[1].text(x=-1.4,y=-1.5,s='AG',fontsize=fontsize,color="#e9dac4",weight="semibold")
axs[1].text(x=-1.4,y=-1.75,s='AT',fontsize=fontsize,color="#749eb2",weight="semibold")

axs[1].text(x=1,y=-1.25,s='ACG',fontsize=fontsize,color="#bd8e83",weight="semibold")
axs[1].text(x=1,y=-1.5,s='AGC',fontsize=fontsize,color="#e9dac4",weight="semibold")
axs[1].text(x=1,y=-1.75,s='ATC',fontsize=fontsize,color="#749eb2",weight="semibold")

axs[1].text(x=1,y=1,s='ATCG',fontsize=fontsize,color="#749eb2",weight="semibold")
axs[1].text(x=1,y=1.25,s='AGCT',fontsize=fontsize,color="#e9dac4",weight="semibold")
axs[1].text(x=1,y=1.5,s='ACGT',fontsize=fontsize,color="#bd8e83",weight="semibold")

axs[1].set(
    xlabel="",ylabel="",
    xticks=[],yticks=[]
)


axs[1].set(xlim=(-1,1),ylim=(-1,1))
plt.tight_layout()
plt.savefig("combined_groupings.png",dpi=300)
