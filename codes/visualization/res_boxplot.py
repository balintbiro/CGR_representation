import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("data/res_iter.csv")

fig,axs=plt.subplots(1)
sns.boxplot(data=df,x="resolution",y="accuracy",hue="dataset",flierprops={"marker":"o"})
plt.tight_layout()
plt.savefig("data/res_iter_new.png",dpi=300)
