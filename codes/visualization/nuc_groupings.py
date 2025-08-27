import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.figure(figsize=(3,3))
# Create the graph
G = nx.Graph()
G.add_edge("A", "C")
G.add_edge("A", "G")
G.add_edge("A", "T")

# Define positions manually to allow for curvature
pos = nx.spring_layout(G)  # You can also define custom positions

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_color="#000000")
nx.draw_networkx_labels(G, pos, font_color="#ffffff")

kwargs={"linestyle":"--"}

connection_width=8
# Draw curved edge A-G using connectionstyle
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "C")],
    edge_color="#bd8e83",
    arrows=True,
    width=connection_width
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("G", "T")],
    edge_color="#bd8e83",
    arrows=True,
    style="--",
    width=connection_width
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "G")],
    edge_color="#e9dac4",
    arrows=True,
    width=connection_width
    #connectionstyle="arc3,rad=.3"  # Adjust rad for curvature
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("C", "T")],
    edge_color="#e9dac4",
    arrows=True,
    style="--",
    width=connection_width,
    label="Strong"
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("A", "T")],
    edge_color="#749eb2",
    arrows=True,
    width=connection_width
    #connectionstyle="arc3,rad=.3"  # Adjust rad for curvature
)
nx.draw_networkx_edges(
    G, pos,
    edgelist=[("C", "G")],
    edge_color="#749eb2",
    arrows=True,
    style="--",
    width=connection_width
)


legend_elements = [
    Line2D([0], [0], color='#bd8e83', lw=0, marker='o', label='Side Group'),
    Line2D([0], [0], color='#e9dac4', lw=0, marker='o', label='Structure'),
    Line2D([0], [0], color='#749eb2', lw=0, marker='o', label='H-bond')
]

plt.legend(handles=legend_elements, loc="best")#bbox_to_anchor=(1,0.75)


plt.axis("off")
plt.tight_layout()
plt.savefig("nuc_groupings.png")
