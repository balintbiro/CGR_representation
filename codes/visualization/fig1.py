from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph

class fig1:
    def __init__(self):
        self.sequence="MHLLGPWLLLLVLEYLAFSDSSKWVFEHPETLYAWEGACVWIPCTYRALDGDLESFILFHNPEYNKNTSKFDGTRLYESTKDGKVPSEQKRVQFLGDKNKNCTLSIHPVHLNDSGQLGLRMESKTEKWMERIHLNVSERPFPPHIQLPPEIQESQEVTLTCLLNFSCYGYPIQLQWLLEGVPMRQAAVTSTSLTIKSVFTRSELKFSPQWSHHGKIVTCQLQDADGKFLSNDTVQLNVKHTPKLEIKVTPSDAIVREGDSVTMTCEVSSSNPEYTTVSWLKDGTSLKKQNTFTLNLREVTKDQSGKYCCQVSNDVGPGRSEEVFLQVQYAPEPSTVQILHSPAVEGSQVEFLCMSLANPLPTNYTWYHNGKEMQGRTEEKVHIPKILPWHAGTYSCVAENILGTGQRGPGAELDVQYPPKKVTTVIQNPMPIREGDTVTLSCNYNSSNPSVTRYEWKPHGAWEEPSLGVLKIQNVGWDNTTIACAACNSWCSWASPVALNVQYAPRDVRVRKIKPLSEIHSGNSVSLQCDFSSSHPKEVQFFWEKNGRLLGKESQLNFDSISPEDAGSYSCWVNNSIGQTASKAWTLEVLYAPRRLRVSMSPGDQVMEGKSATLTCESDANPPVSHYTWFDWNNQSLPYHSQKLRLEPVKVQHSGAYWCQGTNSVGKGRSPLSTLTVYYSPETIGRRVAVGLGSCLAILILAICGLKLQRRWKRTQSQQGLQENSSGQSFFVRNKKVRRAPLSEGPHSLGCYNPMMEDGISYTTLRFPEMNIPRTGDAESSEMQRPPPDCDDTVTYSALHKRQVGDYENVIPDFPEDEGIHYSELIQFGVGERPQAQENVDYVILKH"

    def cgr(self,ax):
        aa_order=list("ACDEFGHIKLMNPQRSTVWY")
        angles=np.linspace(0, 2 * np.pi, len(aa_order), endpoint=False)
        vertices=np.column_stack((np.cos(angles), np.sin(angles)))
        vertex_map = {aa: vertices[i] for i, aa in enumerate(aa_order)}
        coords = np.zeros((len(self.sequence), 2), dtype=float)
        start_point=(0.0,0.0)
        current = np.array(start_point, dtype=float)
        for i, aa in enumerate(self.sequence):
            target = vertex_map[aa]
            current = current + 0.865 * (target - current)
            coords[i] = current
        sns.scatterplot(x=coords[:,0],y=coords[:,1],ax=ax,s=10, c="black")
        radius=1.1
        for key, value in vertex_map.items():
            ax.text(x=radius*value[0], y=radius*value[1], s=key,ha="center",va="center")
        ax.set_aspect("equal")
        ax.set_axis_off()
        return coords

    def fcgr(self,coordinates,ax):
        fcgr, _, _=np.histogram2d(
            coordinates[:,0],coordinates[:,1],
            bins=35,
            range=[[-1.0, 1.0], [-1.0, 1.0]]
        )
        sns.heatmap(
            np.flipud(fcgr.T),
            cmap="gray_r",
            cbar_kws={"shrink": .5, "label": "Frequency"},
            linewidths=.01,
            linecolor="lightgray",
            ax=ax
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
    
    def plot_dashboard(self):
        mosaic=[['A','B']]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(constants.FIG_WIDTH,3),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[1],
                "width_ratios":[1,1]
            }
        )
        coordinates=self.cgr(ax_dict['A'])
        self.fcgr(coordinates,ax_dict['B'])
        ph.label_panels_mosaic(fig, ax_dict)
        return fig,ax_dict

Path.mkdir(Path(constants.FIGURES),parents=True,exist_ok=True)

figure=fig1()
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig1.png",dpi=300)