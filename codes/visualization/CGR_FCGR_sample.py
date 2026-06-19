from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def protein_to_cgr(sequence, aa_order=None, scale=0.5, start_point=(0.0, 0.0)):
    if aa_order is None:
        aa_order="ACDEFGHIKLMNPQRSTVWY"
    aa_order = list(aa_order)
    if len(aa_order) != 20 or len(set(aa_order)) != 20:
        raise ValueError("aa_order must contain exactly 20 unique amino acids.")
    sequence = sequence.upper()
    invalid = set(sequence) - set(aa_order)
    if invalid:
        raise ValueError(f"Invalid amino acids found in sequence: {sorted(invalid)}")
    # Create vertices on a unit circle (regular 20-gon)
    angles = np.linspace(0, 2 * np.pi, len(aa_order), endpoint=False)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    # Amino acid -> vertex mapping
    vertex_map = {aa: vertices[i] for i, aa in enumerate(aa_order)}
    # Iterate CGR
    coords = np.zeros((len(sequence), 2), dtype=float)
    current = np.array(start_point, dtype=float)
    for i, aa in enumerate(sequence):
        target = vertex_map[aa]
        current = current + scale * (target - current)
        coords[i] = current
    return coords, vertex_map

def fcgr(cgr:np.ndarray,resolution:int)->np.ndarray:
    fcgr, _, _ = np.histogram2d(
        cgr[:,0],cgr[:,1],
        bins=resolution,
        range=[[-1.0, 1.0], [-1.0, 1.0]]
    )
    return fcgr.T

bcr_seq="MHLLGPWLLLLVLEYLAFSDSSKWVFEHPETLYAWEGACVWIPCTYRALDGDLESFILFHNPEYNKNTSKFDGTRLYESTKDGKVPSEQKRVQFLGDKNKNCTLSIHPVHLNDSGQLGLRMESKTEKWMERIHLNVSERPFPPHIQLPPEIQESQEVTLTCLLNFSCYGYPIQLQWLLEGVPMRQAAVTSTSLTIKSVFTRSELKFSPQWSHHGKIVTCQLQDADGKFLSNDTVQLNVKHTPKLEIKVTPSDAIVREGDSVTMTCEVSSSNPEYTTVSWLKDGTSLKKQNTFTLNLREVTKDQSGKYCCQVSNDVGPGRSEEVFLQVQYAPEPSTVQILHSPAVEGSQVEFLCMSLANPLPTNYTWYHNGKEMQGRTEEKVHIPKILPWHAGTYSCVAENILGTGQRGPGAELDVQYPPKKVTTVIQNPMPIREGDTVTLSCNYNSSNPSVTRYEWKPHGAWEEPSLGVLKIQNVGWDNTTIACAACNSWCSWASPVALNVQYAPRDVRVRKIKPLSEIHSGNSVSLQCDFSSSHPKEVQFFWEKNGRLLGKESQLNFDSISPEDAGSYSCWVNNSIGQTASKAWTLEVLYAPRRLRVSMSPGDQVMEGKSATLTCESDANPPVSHYTWFDWNNQSLPYHSQKLRLEPVKVQHSGAYWCQGTNSVGKGRSPLSTLTVYYSPETIGRRVAVGLGSCLAILILAICGLKLQRRWKRTQSQQGLQENSSGQSFFVRNKKVRRAPLSEGPHSLGCYNPMMEDGISYTTLRFPEMNIPRTGDAESSEMQRPPPDCDDTVTYSALHKRQVGDYENVIPDFPEDEGIHYSELIQFGVGERPQAQENVDYVILKH"
cgr,vertices=protein_to_cgr(sequence=bcr_seq,scale=0.865)

fcgr=fcgr(cgr=cgr,resolution=35)

fig, axs = plt.subplots(ncols=2, figsize=(8, 5), constrained_layout=True)

# --- left: CGR ---
sns.scatterplot(x=cgr[:, 0], y=cgr[:, 1], ax=axs[0], s=10, c="black")
radius = 1.1
for key, value in vertices.items():
    axs[0].text(x=radius * value[0], y=radius * value[1], s=key)

#axs[0].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
axs[0].axis("off")
axs[0].set_aspect("equal")

# --- right: FCGR ---
heatmap = sns.heatmap(
    np.flipud(fcgr),
    cmap="gray_r",
    cbar_kws={"shrink": .5, "label": "Frequency"},
    linewidths=.01,
    #linecolor="lightgray",
    ax=axs[1]
)

cbar = heatmap.collections[0].colorbar
cbar.locator = MaxNLocator(integer=True)
cbar.update_ticks()

axs[1].set_aspect("equal")
axs[1].tick_params(axis="y", labelrotation=0)
axs[1].set(
    xlabel="Resolution=35",
    ylabel="Resolution=35"
)

# --- IMPORTANT: draw first so final axes positions are known ---
fig.canvas.draw()

# Get axes positions in figure coordinates
bbox0 = axs[0].get_position()
bbox1 = axs[1].get_position()

# Add panel labels in figure coordinates
fig.text(bbox0.x0, bbox0.y1 + 0.01, "A",
         ha="left", va="bottom", fontsize=16)

fig.text(bbox1.x0, bbox1.y1 + 0.01, "B",
         ha="left", va="bottom", fontsize=16)
fig.subplots_adjust(wspace=0.2)
#plt.tight_layout()
plt.savefig("CGR_FCGR.png",dpi=400)