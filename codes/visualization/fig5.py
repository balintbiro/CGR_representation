import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph
from scipy import stats
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.cm import ScalarMappable

from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

class fig5:
    def __init__(self,data,pvals):
        self.data=data
        self.pvals=pvals
        self.boxprops={
            "facecolor":"none",
        }
        self.width=.5
        self.order=[0.9,0.8,0.7,0.6,0.5]
        self.palette={
            "Min":"#5e89ab",
            "Q1":"#df5441",
            "Q2":"#f0a967",
            "Q3":"#070807",
            "Max":"#016e0e"
        }
        self.pairs=[
            (0.9,0.8),
            (0.9,0.7),
            (0.9,0.6),
            (0.9,0.5),
            (0.9,0.4),
            (0.9,0.3),
            (0.8,0.7),
            (0.8,0.6),
            (0.8,0.5),
            (0.8,0.4),
            (0.8,0.3),
            (0.7,0.6),
            (0.7,0.5),
            (0.7,0.4),
            (0.7,0.3),
            (0.6,0.5),
            (0.6,0.4),
            (0.6,0.3),
            (0.5,0.4),
            (0.5,0.3),
            (0.4,0.3)
        ]
        self.pairs=[
            (0.9,0.8),
            (0.9,0.7),
            (0.9,0.6),
            (0.9,0.5)
        ]

    def complex_boxplot(self,data,ax):
        sns.boxplot(
            data=data,x="ODR",y="F1 score",
            ax=ax,
            showfliers=False,
            width=self.width,
            showcaps=False,
            zorder=4,
            boxprops=self.boxprops,
            order=self.order,
            hue_order=self.order
        )
        sns.stripplot(
            data=data,x="ODR",y="F1 score",
            ax=ax,
            alpha=.9,
            zorder=0,
            size=2,
            jitter=.2,
            order=self.order,
            hue_order=self.order,
            color="black"
        )
        annotator=Annotator(ax, self.pairs, data=data, x="ODR", y="F1 score",order=self.order)
        annotator.configure(test='Kruskal', text_format='star', loc='inside',comparisons_correction="fdr_bh",line_width=1,fontsize=constants.TINY_SIZE,hide_non_significant=True)
        annotator.apply_and_annotate()

    def heatmap(self,data,ax,cbar=True):
        cmap=ListedColormap([
            "#d9d9d9",
            "#FFB8B8",
            "#D10000",
            "#470000"
        ])
        norm=BoundaryNorm(
            boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
            ncolors=cmap.N
        )
        sns.heatmap(
            data=data[[1,0.9,0.8,0.7,0.6,0.5]],
            ax=ax,
            cmap=cmap,
            norm=norm,
            cbar=False
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            va="center"
        )
    
    def lineplot(self,data,ax):
        sns.lineplot(
            data=data,
            x="ODR",hue="Rank",y="F1 score",
            ax=ax,err_style="bars",
            palette=self.palette
        )
        ax.set_xticks([1.0,0.9,0.8,0.7,0.6,0.5])
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_xlim(1.05, 0.45)

    def plot_dashboard(self):
        mosaic=[
            ['A','B','C'],
            ['G','H','I'],
            ['D','E','F'],
            ['J','K','L']
        ]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(constants.FIG_WIDTH,6),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[1,0.5,1,0.5],
                "width_ratios":[1,1,1]
            }
        )
        cols=[('A','D'),('B','E'),('C','F')]
        datasets=["DeepLoc","PFAM","ImmunoDB"]
        for index,dataset in enumerate(datasets):
            col=cols[index]
            selection=self.data[(self.data["Dataset"]==dataset)&(self.data["Rank"]!="Mix")]
            self.lineplot(data=selection[selection["Model"]=="Custom"],ax=ax_dict[col[0]])
            self.lineplot(data=selection[selection["Model"]=="ResNet18"],ax=ax_dict[col[1]])
            if index==0:
                ax_dict[col[0]].set_ylabel("Custom\nF1 score")
                ax_dict[col[1]].set_ylabel("ResNet18\nF1 score")
            else:
                ax_dict[col[0]].set_ylabel(None)
                ax_dict[col[1]].set_ylabel(None)
            ax_dict[col[0]].set_title(datasets[index])
        cols=[('G','J'),('H','K'),('I','L')]
        for index,dataset in enumerate(datasets):
            col=cols[index]
            selection=self.pvals[(self.pvals["Dataset"]==dataset)].set_index("Rank")
            if index!=2:
                self.heatmap(data=selection[selection["Model"]=="Custom"],ax=ax_dict[col[0]],cbar=False)
                self.heatmap(data=selection[selection["Model"]=="ResNet18"],ax=ax_dict[col[1]],cbar=False)
            else:
                self.heatmap(data=selection[selection["Model"]=="Custom"],ax=ax_dict[col[0]])
                self.heatmap(data=selection[selection["Model"]=="ResNet18"],ax=ax_dict[col[1]])
            if index==0:
                ax_dict[col[0]].set_ylabel("Custom\nEncodings")
                ax_dict[col[1]].set_ylabel("ResNet18\nEncodings")
            else:
                ax_dict[col[0]].set_ylabel(None)
                ax_dict[col[0]].set_yticklabels([])
                ax_dict[col[1]].set_ylabel(None)
                ax_dict[col[1]].set_yticklabels([])

        handles, labels=ax_dict["A"].get_legend_handles_labels()
        for ax in ax_dict.values():
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        fig.legend(
            handles,
            labels,
            title="Encodings",
            loc="lower center",
            ncols=4,
            bbox_to_anchor=(0.55,-0.075),
            frameon=False
        )
        ph.label_panels_mosaic(fig, ax_dict,cols2leave=list("GHIJKL"))
        cmap=ListedColormap([
            "#d9d9d9",
            "#FFB8B8",
            "#D10000",
            "#470000"
        ])
        cax=fig.add_axes([0.425, -0.15, 0.25, 0.03])
        norm=BoundaryNorm(
            boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
            ncolors=cmap.N
        )
        sm=ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar=fig.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            ticks=[0,1,2,3]
        )
        cbar.set_ticklabels([
            "ns",
            "*",
            "**",
            "***"
        ])
        cbar.ax.tick_params(length=0)
        cbar.ax.set_title("Significance")
        cbar.outline.set_visible(True)
        return fig,ax_dict

results=pd.read_csv(constants.RESULTS/"augmentation_results.csv")
results.rename(columns={"f1":"F1 score","dataset":"Dataset","rank":"Rank","model":"Model","odr":"ODR"},inplace=True)
results["Dataset"]=results["Dataset"].replace(["Deeploc","Pfam","Immune"],["DeepLoc","PFAM","ImmunoDB"])
results["Model"]=results["Model"].replace(["resnet","custom"],["ResNet18","Custom"])
results["Rank"]=results["Rank"].str.capitalize()

cv_results=pd.read_csv(constants.RESULTS/"cv_results.csv")
cv_results.rename(columns={"f1":"F1 score","dataset":"Dataset","rank":"Rank","model":"Model"},inplace=True)
cv_results["Dataset"]=cv_results["Dataset"].replace(["deeploc","pfam","immune"],["DeepLoc","PFAM","ImmunoDB"])
cv_results["Model"]=cv_results["Model"].replace(["resnet","custom"],["ResNet18","Custom"])
cv_results["Rank"]=cv_results["Rank"].str.capitalize()
cv_results=cv_results[cv_results["Rank"]=="Min"]
cv_results["ODR"]=1.0
container=[]
dedicated_encodings=["Q1","Q2","Q3","Max"]
for i in range(len(dedicated_encodings)):
    copied=cv_results.copy()
    copied["Rank"]=dedicated_encodings[i]
    container.append(copied)
cv_results=pd.concat(container)

results=pd.concat([cv_results,results])

pvals=[]
for dataset in results["Dataset"].unique():
    for model in results["Model"].unique():
        for rank in ["Q1","Q2","Q3","Max"]:
            selection=results[
                (results["Dataset"]==dataset)
                &(results["Model"]==model)
                &(results["Rank"]==rank)
            ]
            for odr in [1,0.9,0.8,0.7,0.6,0.5]:
                a=selection[selection["ODR"]==1]["F1 score"]
                b=selection[selection["ODR"]==odr]["F1 score"]
                pval=stats.kruskal(a,b).pvalue
                pvals.append([dataset,model,rank,odr,pval])
pvals=pd.DataFrame(data=pvals,columns=["Dataset","Model","Rank","ODR","pvalue"])

_, pvals["pvalue"], _, _ = multipletests(
    pvals["pvalue"],
    method="fdr_bh"
)
pvals["pvalue"]=np.select(
    [
        pvals["pvalue"].values>0.05,
        (pvals["pvalue"].values<0.05)&(pvals["pvalue"]>0.01),
        (pvals["pvalue"].values<0.01)&(pvals["pvalue"]>0.001),
        pvals["pvalue"].values<0.001
    ],
    [0,1,2,3]
)

pvals_wide=(
    pvals
    .pivot(
        index=["Dataset", "Model", "Rank"],
        columns="ODR",
        values="pvalue"
    )
    .reset_index()
)


figure=fig5(data=results,pvals=pvals_wide)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig5.png",dpi=300,bbox_inches="tight")
