import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition

oe = np.loadtxt("GSM17_100koe.mat")
cor_oe = np.corrcoef(np.cov(oe))
cor_oe[np.isnan(cor_oe)] = 0

from sklearn.decomposition import PCA
# use pca get pc1
pca = PCA(n_components=2)
pcaOut = pca.fit_transform(cor_oe)
pcaOut = pd.DataFrame(pcaOut)
pcaOut.columns = ["pc1","pc2"]
# pcaOut['lin'] = np.arange(1,1876)
#print (pcaOut["pc1"])
kmeans = KMeans(n_clusters=2).fit(pcaOut)
plt.figure(figsize=(8,8))
# plt.subplot(221)
plt.scatter(pcaOut["pc1"], pcaOut["pc2"], c=kmeans.labels_, cmap=plt.cm.Set1)
# plt.subplot(222)

plt.show()