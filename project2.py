import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.externals.joblib import Memory
from sklearn.model_selection import GridSearchCV
from shutil import rmtree
from tempfile import mkdtemp

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
X_svd = svd.fit_transform(X_tfidf)
percent_var = np.cumsum(svd.explained_variance_ratio_)
plt.plot(range(1,n_components+1), percent_var)
plt.xlabel('r')
plt.ylabel('variance retained')
#plt.show(0)

""" 
Question 5: 
Let 𝑟 be the dimension that we want to reduced the data to (i.e. n_components). 
Try  𝑟=1,2,3,5,10,20,50,100,300  and plot the five measure scores vs  𝑟  for both SVD and NMF. 
Report the best 𝑟 choice for SVD and NMF, respectively.
"""

# Helper function to calculate score metrics. Returns a list with all the scores.
def calculate_scores(y_true, y_pred):
  homogeneity = metrics.homogeneity_score(y_true, y_pred)
  completeness = metrics.completeness_score(y_true, y_pred)
  v_measure = metrics.v_measure_score(y_true, y_pred)
  adj_rand = metrics.adjusted_rand_score(y_true, y_pred)
  adj_mutual_info = metrics.adjusted_mutual_info_score(y_true, y_pred)
  return [homogeneity, completeness, v_measure, adj_rand, adj_mutual_info]

# Metrics for reduced by SVD data
# We've already done SVD for the first 1000 components, i.e. we can get the
# lower r values from X_svd with n_components=1000
r = [1, 2, 3, 5, 10, 20, 50, 100, 300]
scores_svd = []

for n_comp in r:
  km.fit(X_svd[:, :n_comp])
  
  # Calculate metrics
  scores_svd.extend(calculate_scores(labels, km.labels_))

# Plot the data
scores = ['homegeneity', 'completeness', 'v-measure', 'adjusted rand', 
           'adjusted mutual info']
scores_svd_array = np.asarray(scores_svd).reshape(len(r),5)
#plt.plot(r, scores_svd_array)
#plt.legend(legend)

# Create pandas dataframe
scores_svd_df = pd.DataFrame(data=scores_svd_array, index=r, columns=scores)
scores_svd_df.plot()
plt.title('K-Mean scores over data reduced by SVD')
plt.xlabel('$r$')
plt.ylabel('Score')
plt.xscale('log')
plt.show()
print('\nScores:')
print(scores_svd_df)

# Metrics for data reduced by NMF
# Perform k-means clustering
scores_nmf = []
for n_comp in r:
  X_nmf = NMF(n_components=n_comp).fit_transform(X_tfidf)
  km.fit(X_nmf)
  
  # Calculate metrics
  scores_nmf.extend(calculate_scores(labels, km.labels_))

# Plot results
scores_nmf_array = np.asarray(scores_nmf).reshape(len(r),5)
scores_nmf_df = pd.DataFrame(data=scores_nmf_array, index=r, columns=scores)
#plt.plot(r, scores_nmf_array)
#plt.legend(legend)
scores_nmf_df.plot()
plt.title('K-Mean Scores versus Data Dimensions')
plt.xlabel('$r$')
plt.ylabel('Score')
plt.xscale('log')
plt.show()
print('\nScores:')
print(scores_nmf_df)

"""
Question 7: Visualization
"""
### SVD with its best 'r' value
# Reduced the term matrix with r=n_components=2 with SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_tfidf)

# Use KMeans to find 2 clusters (kmeans defined in question 2) and predict labels
km.fit(X_svd)
y_svd = km.predict(X_svd)

# Plot ground truth labels
plt.scatter(X_svd[:,0], X_svd[:,1], c=labels)
plt.title('Clusters with ground truth labels for SVD-reduced ($r=2$)')
plt.show()

# Plot clusters with clustering labels as colors
plt.scatter(X_svd[:,0], X_svd[:,1], c=y_svd)
plt.title('Clusters with predicted labels for SVD-reduced data ($r=2$)')
plt.show()

### NMF with its best 'r' value
# Reduce the term matrix with r=3 (need to confirm this is the best)
nmf = NMF(n_components=2)
X_nmf = nmf.fit_transform(X_tfidf)

# Find clusters and predict labels
km.fit(X_nmf)
y_nmf = km.predict(X_nmf)

# Plot ground truth labels
plt.scatter(X_nmf[:,0], X_nmf[:,1], c=labels)
plt.title('Clusters with ground truth labels for NMF-reduced data ($r=2$)')
plt.show()

# Plot clusters with predicted labels
plt.scatter(X_nmf[:,0], X_nmf[:,1], c=y_svd)
plt.title('Clusters with predicted labels for NMF-reduced data ($r=2$)')
plt.show()

"""
In this part we want to examine how purely we can retrieve all 20 original sub-class labels
with clustering. Therefore, we need to include all the documents and the corresponding
terms in the data matrix and find proper representation through dimensionality reduction
of the TF-IDF representation.

QUESTION 11: Repeat the following for 20 categories using the same parameters as in
2-class case, but with k=20:
• Transform corpus to TF-IDF matrix;
• Directly perform K-means and report the 5 measures and the contingency matrix;

QUESTION 12: Try different dimensions for both truncated SVD and NMF dimensionality
reduction techniques and the different transformations of the obtained feature vectors as
outlined in above parts.
You don’t need to report everything you tried, which will be tediously long. You are asked,
however, to report your best combination, and quantitatively report how much better
it is compared to other combinations. You should also include typical combinations
showing what choices are desirable (or undesirable).

"""

"""
Question 11
"""
dataset = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
labels = dataset.target
vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
X_tfidf = vectorizer.fit_transform(dataset.data)
km = KMeans(n_clusters=20, random_state=0, n_init=30, max_iter=1000)
km.fit(X_tfidf)
print(f"Homogeneity: {metrics.homogeneity_score(labels, km.labels_):.3f}")
print(f"Completeness: {metrics.completeness_score(labels, km.labels_):.3f}")
print(f"V-measure: {metrics.v_measure_score(labels, km.labels_):.3f}")
print(f"Adjusted Rand-Index: {metrics.adjusted_rand_score(labels, km.labels_):.3f}")
print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels, km.labels_, 'arithmetic')}")

"""
Question 12
"""
cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=10)
pipeline = Pipeline([
  ('vect', TfidfVectorizer(min_df=3, stop_words='english')),
  ('reduce_dim', TruncatedSVD()),
  ('clf', KMeans(n_clusters=20, random_state=0, n_init=30, max_iter=1000))
  ]
  ,
  memory=memory
)

param_grid = [
  {
    'vect__min_df': [3, 5],
    'reduce_dim': [TruncatedSVD(), NMF()],
    'reduce_dim__n_components': [2,3,5,10,15,20,25],
  },
]
grid = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_grid, 
                    scoring='adjusted_rand_score')

grid.fit(dataset.data, dataset.target)
result = pd.DataFrame(grid.cv_results_)
print(result)
rmtree(cachedir)
