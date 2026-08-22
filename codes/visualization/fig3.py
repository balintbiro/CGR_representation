import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph

from statannotations.Annotator import Annotator

class fig3:
    def __init__(self,data,cv_results):
        self.data=data
        self.cv_results=cv_results
        self.palette={
            "Min":"#5e89ab",
            "Q1":"#df5441",
            "Q2":"#f0a967",
            "Q3":"#070807",
            "Max":"#016e0e"
        }
        self.boxprops={
            "facecolor":"none",
        }
        self.width=.5
        self.pairs=[
            ("Max","Q3"),
            ("Max","Q2"),
            ("Max","Q1"),
            ("Max","Min"),
            ("Q3","Q2"),
            ("Q3","Q1"),
            ("Q3","Min"),
            ("Q2","Q1"),
            ("Q2","Min"),
            ("Q1","Min")
        ]

    def hist(self,sample_data,ax):
        sample_data.rename(columns={"f1":"F1 score"},inplace=True)
        sns.histplot(
            data=sample_data["F1 score"],
            ax=ax,
            kde=True,
            element="step",
            color="grey"
        )
        sorted=sample_data.sort_values(by="F1 score").reset_index(drop=True).copy()
        ax.axvline(x=sorted.iloc[0]["F1 score"],**{"linestyle":"--","color":"#5e89ab","label":"Min"})
        ax.axvline(x=sorted.iloc[249]["F1 score"],**{"linestyle":"--","color":"#df5441","label":"Q1"})
        ax.axvline(x=sorted.iloc[499]["F1 score"],**{"linestyle":"--","color":"#f0a967","label":"Q2"})
        ax.axvline(x=sorted.iloc[749]["F1 score"],**{"linestyle":"--","color":"#070807","label":"Q3"})
        ax.axvline(x=sorted.iloc[999]["F1 score"],**{"linestyle":"--","color":"#016e0e","label":"Max"})
        ylim=ax.get_ylim()
        ax.set_ylim(ylim[0],ylim[1]*1.25)
        ax.legend(title="Encodings")

    def scatter(self,sample_data,ax):
        sns.scatterplot(
            data=sample_data,
            x="Custom",y="ResNet18",hue="Dataset",
            ax=ax,**{"s":4,"alpha":.4,"edgecolors":None}
        )
        ax.set(ylim=(0.4,0.9),xlim=(0.4,0.9),xlabel="Custom\nF1 score",ylabel="ResNet18\nF1 score")
        ax.plot([0.4,0.9],[0.4,0.9],"k--")
        legend=ax.get_legend()
        if legend is not None:
            for handle in legend.legend_handles:
                handle.set_alpha(1)
                handle.set_markersize(4)

    def complex_boxplot(self,data,ax):
        sns.boxplot(
            data=data,x="Rank",hue="Rank",y="F1 score",
            ax=ax,
            showfliers=False,
            width=self.width,
            showcaps=False,
            zorder=4,
            boxprops=self.boxprops,
            order=["Min","Q1","Q2","Q3","Max"],
            hue_order=["Min","Q1","Q2","Q3","Max"]
        )
        sns.stripplot(
            data=data,x="Rank",hue="Rank",y="F1 score",
            ax=ax,
            alpha=.9,
            zorder=0,
            size=2,
            jitter=.2,
            palette=self.palette,
            order=["Min","Q1","Q2","Q3","Max"],
            hue_order=["Min","Q1","Q2","Q3","Max"]
        )
        annotator=Annotator(ax, self.pairs, data=data, x="Rank", y="F1 score",order=["Min","Q1","Q2","Q3","Max"])
        annotator.configure(test='Kruskal', text_format='star', loc='inside',comparisons_correction="fdr_bh",line_width=1,fontsize=constants.TINY_SIZE,hide_non_significant=True)
        annotator.apply_and_annotate()

    def plot_dashboard(self):
        mosaic=[
            ['A','B','C','D'],
            ['E','F','G','H']
        ]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(constants.FIG_WIDTH,4),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[1,1],
                "width_ratios":[2,1,1,1]
            }
        )
        share_group=list("BCDFGH")
        for label in share_group[1:]:
            ax_dict[label].sharex(ax_dict[share_group[0]])
        hist_selection=self.data[(self.data["model"]=="ResNet18")&(self.data["dataset"]=="PFAM")]
        self.hist(sample_data=hist_selection,ax=ax_dict['A'])
        scatter_selection=results[search["mode"]=="Search"].pivot_table(
            index=["encoding", "task", "dataset", "mode"],
            columns="model",
            values="f1"
        ).reset_index()
        scatter_selection.rename(columns={"dataset":"Dataset"},inplace=True)
        self.scatter(sample_data=scatter_selection,ax=ax_dict['E'])
        cols=[('B','F'),('C','G'),('D','H')]
        datasets=["DeepLoc","PFAM","ImmunoDB"]
        for index,dataset in enumerate(datasets):
            col=cols[index]
            selection=self.cv_results[self.cv_results["Dataset"]==dataset]
            self.complex_boxplot(data=selection[selection["Model"]=="Custom"],ax=ax_dict[col[0]])
            self.complex_boxplot(data=selection[selection["Model"]=="ResNet18"],ax=ax_dict[col[1]])
            if index==0:
                ax_dict[col[0]].set_ylabel("Custom\nF1 score")
                ax_dict[col[1]].set_ylabel("ResNet18\nF1 score")
            else:
                ax_dict[col[0]].set_ylabel(None)
                ax_dict[col[1]].set_ylabel(None)
            ax_dict[col[0]].set_title(datasets[index])
            ax_dict[col[1]].set_xlabel("Encodings")
        ph.label_panels_mosaic(fig, ax_dict)
        return fig,ax_dict

datasets=["deeploc","pfam","immune"]
dfs=[]
for dataset in datasets:
    search=pd.read_csv(constants.RESULTS/f"{dataset}_search.csv")
    search["mode"]="Search"
    dfs.append(search)

results=pd.concat(dfs)
results["model"].replace(["resnet","custom"],["ResNet18","Custom"],inplace=True)
results["dataset"].replace(["deeploc","pfam","immune"],["DeepLoc","PFAM","ImmunoDB"],inplace=True)

cv_results=pd.read_csv(constants.RESULTS/"cv_results.csv")
cv_results.rename(columns={"f1":"F1 score","dataset":"Dataset","rank":"Rank","model":"Model"},inplace=True)
cv_results["Dataset"].replace(["deeploc","pfam","immune"],["DeepLoc","PFAM","ImmunoDB"],inplace=True)
cv_results["Model"].replace(["resnet","custom"],["ResNet18","Custom"],inplace=True)
cv_results["Rank"]=cv_results["Rank"].str.capitalize()

figure=fig3(data=results,cv_results=cv_results)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig3.png",dpi=300)
