import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph

from statannotations.Annotator import Annotator

class fig4:
    def __init__(self,data):
        self.data=data
        self.boxprops={
            "facecolor":"none",
        }
        self.width=.5
        self.order=[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3]
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
            (1.0,0.9),
            (1.0,0.8),
            (1.0,0.7),
            (1.0,0.6),
            (1.0,0.5),
            (1.0,0.4),
            (1.0,0.3)
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

    def plot_dashboard(self):
        mosaic=[
            ['A','B','C'],
            ['D','E','F']
        ]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(constants.FIG_WIDTH,4),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[1,1],
                "width_ratios":[1,1,1]
            }
        )
        cols=[('A','D'),('B','E'),('C','F')]
        datasets=["DeepLoc","PFAM","ImmunoDB"]
        for index,dataset in enumerate(datasets):
            col=cols[index]
            selection=self.data[(self.data["Dataset"]==dataset)&(self.data["Rank"]=="Mix")]
            self.complex_boxplot(data=selection[selection["Model"]=="Custom"],ax=ax_dict[col[0]])
            self.complex_boxplot(data=selection[selection["Model"]=="ResNet18"],ax=ax_dict[col[1]])
            if index==0:
                ax_dict[col[0]].set_ylabel("Custom\nF1 score")
                ax_dict[col[1]].set_ylabel("ResNet18\nF1 score")
            else:
                ax_dict[col[0]].set_ylabel(None)
                ax_dict[col[1]].set_ylabel(None)
            ax_dict[col[0]].set_title(datasets[index])
            ax_dict[col[0]].set_xlabel(None)
            ax_dict[col[1]].set_xlabel("ODR")
        ph.label_panels_mosaic(fig, ax_dict)
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
cv_results["Rank"]="Mix"
cv_results["ODR"]=1.0

results=pd.concat([cv_results,results])


figure=fig4(data=results)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig4.png",dpi=300)