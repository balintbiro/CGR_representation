import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import constants
from pathlib import Path
import plot_helpers as ph
from scipy import stats
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class fig6:
    def __init__(self,results,dfs1,dfs2):
        self.results=results
        self.dfs1=dfs1
        self.dfs2=dfs2
        self.order=[1.0,0.9,0.8,0.7,0.6,0.5]
        self.palette={
            "Min":"#5e89ab",
            "Q1":"#df5441",
            "Q2":"#f0a967",
            "Q3":"#070807",
            "Max":"#016e0e"
        }

    def lineplot(self,data,ax):
        sns.lineplot(
            data=data,
            x="ODR",hue="Rank",y="F1 score",
            ax=ax,err_style="bars",
            palette=self.palette,
            legend=False
        )
        ax.set_xticks([1.0,0.9,0.8,0.7,0.6,0.5])
        ax.set_xticklabels(["1.0","0.9","0.8","0.7","0.6","0.5"])
        ax.set_xlabel("ODR")
        ax.set_xlim(1.05, 0.45)

    def scatterplot(self,x,y,ax,label):
        plot=sns.scatterplot(
            x=x,y=y,hue=label,
            ax=ax,
            s=10,alpha=0.15,
            palette=dict(Augmented="skyblue",Original="lightpink"),
            **{"linewidths":0}
        )
        return plot

    def confidence_ellipse(self,x, y, ax, n_std=2.0,  **kwargs):
        x=np.asarray(x)
        y=np.asarray(y)

        if x.ndim!=1 or y.ndim != 1:
            raise ValueError("x and y must be 1D")
        if x.size < 2:
            raise ValueError("Need at least 2 points")

        cov=np.cov(x, y)
        vals, vecs=np.linalg.eigh(cov)

        order=vals.argsort()[::-1]
        vals=vals[order]
        vecs=vecs[:, order]

        angle=np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height=2 * n_std * np.sqrt(vals)
        ellipse=Ellipse(
            (np.mean(x), np.mean(y)),
            width=width,
            height=height,
            angle=angle,
            **kwargs
        )
        ax.add_patch(ellipse)
        return ellipse

    def augment(
            self,
            original_data:pd.DataFrame,
            to_add:pd.DataFrame,
            original_ratio:float
        )->pd.DataFrame:
        augmented_size=int(original_data.shape[0]/original_ratio)
        sample_size=int(augmented_size-original_data.shape[0])
        original_data["Label"]="Original"
        sampled=to_add.sample(n=sample_size,random_state=0)
        sampled["Label"]="Augmented"
        augmented=pd.concat(
            [
                original_data,
                sampled
            ]
        )
        return augmented

    def plot_dashboard(self):
        mosaic=[
            ['A','A','A','B','B','B'],
            ['C','I','E','F','G','H'],
            ['D','J','K','L','M','N']
        ]
        fig,ax_dict=plt.subplot_mosaic(
            mosaic,
            figsize=(constants.FIG_WIDTH,4),
            dpi=400,
            constrained_layout=True,
            gridspec_kw={
                "height_ratios":[0.5,1,1],
                "width_ratios":[1,1,1,1,1,1]
            }
        )
        ex1=self.results[
            (self.results["Dataset"]=="DeepLoc")
            &(self.results["Model"]=="Custom")
            &(self.results["Rank"]=="Q1")
        ]
        ex2=self.results[
            (self.results["Dataset"]=="PFAM")
            &(self.results["Model"]=="ResNet18")
            &(self.results["Rank"]=="Q2")
        ]
        self.lineplot(data=ex1,ax=ax_dict['A'])
        self.lineplot(data=ex2,ax=ax_dict['B'])
        ax_dict['B'].set_ylabel(None)
        cols=[('C','D'),('I','J'),('E','K'),('F','L'),('G','M'),('H','N')]
        for index,ratio in enumerate(self.order):
            col=cols[index]
            augmented1=self.augment(
                original_data=self.dfs1[0],
                to_add=self.dfs1[1],
                original_ratio=ratio
            ).reset_index(drop=True)
            augmented2=self.augment(
                original_data=self.dfs2[0],
                to_add=self.dfs2[1],
                original_ratio=ratio
            ).reset_index(drop=True)
            transformed1=augmented1.drop(columns=["Label"]).div(augmented1.drop(columns=["Label"]).max(axis=1),axis=0)
            transformed2=augmented2.drop(columns=["Label"]).div(augmented2.drop(columns=["Label"]).max(axis=1),axis=0)
            pca=PCA(n_components=2)
            dims1=pd.DataFrame(pca.fit_transform(transformed1),columns=["dim1","dim2"])
            dims1["Label"]=augmented1["Label"].values
            dims2=pd.DataFrame(pca.fit_transform(transformed2),columns=["dim1","dim2"])
            dims2["Label"]=augmented2["Label"].values
            colors=["skyblue","lightpink"]
            sp1=self.scatterplot(x=dims1.dim1,y=dims1.dim2,ax=ax_dict[col[0]],label=augmented1["Label"])
            if index!=0:
                self.confidence_ellipse(x=dims1[dims1["Label"]=="Original"].dim1,y=dims1[dims1["Label"]=="Original"].dim2,ax=ax_dict[col[0]],n_std=3,edgecolor=colors[1],facecolor=mcolors.to_rgba(colors[1],alpha=0.1))
                self.confidence_ellipse(x=dims1[dims1["Label"]=="Augmented"].dim1,y=dims1[dims1["Label"]=="Augmented"].dim2,ax=ax_dict[col[0]],n_std=3,edgecolor=colors[0],facecolor=mcolors.to_rgba(colors[0],alpha=0.1))
                self.confidence_ellipse(x=dims2[dims2["Label"]=="Original"].dim1,y=dims2[dims2["Label"]=="Original"].dim2,ax=ax_dict[col[1]],n_std=3,edgecolor=colors[1],facecolor=mcolors.to_rgba(colors[1],alpha=0.1))
                self.confidence_ellipse(x=dims2[dims2["Label"]=="Augmented"].dim1,y=dims2[dims2["Label"]=="Augmented"].dim2,ax=ax_dict[col[1]],n_std=3,edgecolor=colors[0],facecolor=mcolors.to_rgba(colors[0],alpha=0.1))
            sp2=self.scatterplot(x=dims2.dim1,y=dims2.dim2,ax=ax_dict[col[1]],label=augmented2["Label"])
            #self.confidence_ellipse(x=dims2.dim1,y=dims2.dim2,ax=ax_dict[col[1]])
            if index==0:
                ax_dict[col[0]].set_ylabel("PC2",loc="bottom")
                ax_dict[col[1]].set_ylabel("PC2",loc="bottom")
                ax_dict[col[0]].set_xlabel("PC1",loc="left")
                ax_dict[col[1]].set_xlabel("PC1",loc="left")

                y0, y1 = ax_dict[col[0]].get_ylim()
                x0, x1 = ax_dict[col[0]].get_xlim()
                ax_dict[col[0]].spines['left'].set_bounds(y0, y0 + 0.3*(y1 - y0))
                ax_dict[col[0]].spines['bottom'].set_bounds(x0, x0 + 0.3*(x1 - x0))

                y0, y1 = ax_dict[col[1]].get_ylim()
                x0, x1 = ax_dict[col[1]].get_xlim()
                ax_dict[col[1]].spines['left'].set_bounds(y0, y0 + 0.3*(y1 - y0))
                ax_dict[col[1]].spines['bottom'].set_bounds(x0, x0 + 0.3*(x1 - x0))
            else:
                ax_dict[col[0]].set_axis_off()
                ax_dict[col[1]].set_axis_off()
            ax_dict[col[0]].set_yticks([])
            ax_dict[col[1]].set_yticks([])
            ax_dict[col[0]].set_xticks([])
            ax_dict[col[1]].set_xticks([])
            ax_dict[col[0]].set_title(f"ODR={ratio}")
            if index!=5:
                sp1.get_legend().remove()
                sp2.get_legend().remove()
            else:
                sp1.get_legend().remove()
                handles,labels=sp2.get_legend_handles_labels()
                for handle in handles:
                    handle.set_alpha(1)
                    if hasattr(handle,"set_sizes"):
                        handle.set_sizes([40])
                    elif hasattr(handle,"set_markersize"):
                        handle.set_markersize(8)
                ax_dict[col[1]].legend(
                    handles,labels,
                    loc="center right",
                    bbox_to_anchor=(1.15,1.2),
                    frameon=True,
                    title=None,
                    handletextpad=0.3
                )

            

        ph.label_panels_mosaic(fig, ax_dict,cols2leave=list("IEFGHJKLMN"))
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

dl_cu_min=pd.read_csv(constants.RESULTS/"dedicated_encodings/deeploc/custom/min.csv")
dl_cu_q1=pd.read_csv(constants.RESULTS/"dedicated_encodings/deeploc/custom/q1.csv")

pf_rn_min=pd.read_csv(constants.RESULTS/"dedicated_encodings/pfam/resnet/min.csv")
pf_rn_q2=pd.read_csv(constants.RESULTS/"dedicated_encodings/pfam/resnet/q2.csv")
    
figure=fig6(
    results=results,
    dfs1=[dl_cu_min,dl_cu_q1],
    dfs2=[pf_rn_min,pf_rn_q2]
)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig6.png",dpi=300,bbox_inches="tight")