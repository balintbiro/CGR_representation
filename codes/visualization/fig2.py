import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph

from statannotations.Annotator import Annotator

class fig2:
    def __init__(self,combined_results):
        self.combined_results=combined_results
        self.palette={
            'Noise':'skyblue',
            'Search':'lightpink'
        }
        self.boxprops={
            "facecolor":"none",
        }
        self.width=.5
        self.pairs=[
            ("Noise","Search")
        ]

    def complex_boxplot(self,data,ax,mask_outlrs=False):
        sns.boxplot(
            data=data,x="mode",hue="mode",y="f1",
            ax=ax,
            showfliers=False,
            width=self.width,
            showcaps=False,
            zorder=4,
            boxprops=self.boxprops
        )
        sns.stripplot(
            data=data,x="mode",hue="mode",y="f1",
            ax=ax,
            alpha=.9,
            zorder=0,
            size=1,
            jitter=.2,
            palette=self.palette
        )
        annotator=Annotator(ax, self.pairs, data=data, x="mode", y="f1")
        annotator.configure(test='Kruskal', text_format='star', loc='inside',comparisons_correction="fdr_bh",line_width=1,fontsize=constants.TINY_SIZE)
        annotator.apply_and_annotate()
        if mask_outlrs:
            q1,q3=np.percentile(data["f1"],[25,75])
            iqr=q3-q1
            ylim=ax.get_ylim()
            ymin,ymax=(q1-(1.5*iqr)),ylim[1]
            ax.set_ylim(ymin,ymax)


    def plot_dashboard(self):
        mosaic=[
            ['A','B','C'],
            ['D','E','F']
        ]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(5,3),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[1,1],
                "width_ratios":[1,1,1]
            },sharex=True
        )
        cols=[('A','D'),('B','E'),('C','F')]
        datasets=["deeploc","pfam","immune"]
        dataset_names=["DeepLoc","PFAM","ImmunoDB"]
        for index,dataset in enumerate(datasets):
            col=cols[index]
            selection=self.combined_results[self.combined_results["dataset"]==dataset]
            self.complex_boxplot(data=selection[selection["model"]=="custom"],ax=ax_dict[col[0]])
            self.complex_boxplot(data=selection[selection["model"]=="resnet"],ax=ax_dict[col[1]],mask_outlrs=True)
            if index==0:
                ax_dict[col[0]].set_ylabel("Custom\nF1 score")
                ax_dict[col[1]].set_ylabel("ResNet18\nF1 score")
            else:
                ax_dict[col[0]].set_ylabel(None)
                ax_dict[col[1]].set_ylabel(None)
            ax_dict[col[0]].set_title(dataset_names[index])
            ax_dict[col[1]].set_xlabel(None)
        ph.label_panels_mosaic(fig, ax_dict)
        return fig,ax_dict

datasets=["deeploc","pfam","immune"]
dfs=[]
for dataset in datasets:
    noise=pd.read_csv(constants.RESULTS/f"{dataset}_noise.csv")
    search=pd.read_csv(constants.RESULTS/f"{dataset}_search.csv")
    noise["mode"]="Noise"
    search["mode"]="Search"
    dfs.append(noise.rename(columns={"type":"model"}))
    dfs.append(search)

results=pd.concat(dfs)

figure=fig2(combined_results=results)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig2.png",dpi=300)