

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition

cor_oe = np.loadtxt("GSM17_100koe.mat")
# cor_oe = np.corrcoef(np.cov(oe))
# cor_oe[np.isnan(cor_oe)] = 0


cmap = sns.diverging_palette(240, 10, as_cmap=True)
ax2 = plt.subplot(2,1,2,frameon=False)
plt.figure(figsize=(25,20))
sns.heatmap(cor_oe,cmap = cmap,center=0,vmax=2,vmin=0,yticklabels=False,xticklabels=150)#

# pca.explained_variance_ratio_
plt.show()