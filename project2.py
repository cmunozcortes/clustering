import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

"""
Loading Dataset
"""
random.seed(42)
np.random.seed(42)

categories = ['comp.sys.ibm.pc.hardware', 'comp.graphics',\
              'comp.sys.mac.hardware', 'comp.os.ms-windows.misc',\
              'rec.autos', 'rec.motorcycles',\
              'rec.sport.baseball', 'rec.sport.hockey']

dataset = fetch_20newsgroups(subset='all', categories=categories,\
                             shuffle=True, random_state=42)

"""
Question 1: Report the dimensions of the TF-IDF matrix you get. 

"""
vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
X_tfidf = vectorizer.fit_transform(dataset.data)


print(f"Tfidf matrix shape: {X_tfidf.shape}")

"""
Question 2: 

Contingency table A is the matrix whose entries Aij is the number 
of data points that belong to both the class C_i the cluster K_j.
"""
vfunc = np.vectorize(lambda target: target // 4)
labels = vfunc(dataset.target)

km = KMeans(n_clusters=2, random_state=0, n_init=30, max_iter=1000)
km.fit(X_tfidf)
print("Contingent Table") 
print(f"{metrics.cluster.contingency_matrix(labels, km.labels_)}")

""" 
Question 3

Homogeneity is a measure of how “pure” the clusters are. If each cluster 
contains only data points from a single class, the homogeneity is satisfied

Clustering result satisfies completeness if all data points of 
a class are assigned to the same cluster

The V-measure is defined to be the harmonic average of homogeneity score
and completeness score.

The adjusted Rand Index is similar to accuracy measure, which computes
similarity between the clustering labels and ground truth labels.

Adjusted mutual information score measures the mutual information
between the cluster label distribution and the ground truth label distributions.
"""
print(f"Homogeneity: {metrics.homogeneity_score(labels, km.labels_):.3f}")
print(f"Completeness: {metrics.completeness_score(labels, km.labels_):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels, km.labels_):.3f}")
print(f"Adjusted Rand-Index: {metrics.adjusted_rand_score(labels, km.labels_):.3f}")
print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels, km.labels_, 'arithmetic')}")

"""
Question 4
"""
n_components = 1000
svd = TruncatedSVD(n_components=n_components)
svd.fit(X_tfidf)
percent_var = svd.explained_variance_ratio_
plt.plot(range(1,n_components+1), percent_var)
plt.xlabel('r')
plt.ylabel('variance retained')
plt.show(0)
