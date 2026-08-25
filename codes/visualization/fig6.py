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
from matplotlib.patches import Ellipse

def augment(
        original_data:pd.DataFrame,
        to_add:pd.DataFrame,
        original_ratio:float
    )->pd.DataFrame:
    augmented_size=int(original_data.shape[0]/original_ratio)
    sample_size=int(augmented_size-original_data.shape[0])
    augmented=pd.concat(
        [
            original_data,
            to_add.sample(n=sample_size,random_state=0)
        ]
    )
    return augmented

def confidence_ellipse(x, y, ax, n_std=2.0,  **kwargs):
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

class fig6:
    def __init__(self,results):
        self.results=results
        self.order=[0.9,0.8,0.7,0.6,0.5]
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

    def scatterplot(self,x,y,ax):
        sns.scatterplot(
            x=x,y=y,ax=ax
        )

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


    
figure=fig6(results=results)
fig,axs=figure.plot_dashboard()
plt.savefig(constants.FIGURES/"fig6.png",dpi=300,bbox_inches="tight")