import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as metric
import sklearn.metrics as metrics
from sklearn import cluster,datasets
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def gau(X,y):
    model =GaussianNB()
    model.fit(X,y)
    y =model.predict(X)
    lim =plt.axis()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='RdBu', alpha=0.1)
    plt.plot(y)
    plt.axis(lim)
    print(model.score(X,y))
    plt.show()
def kmeans(X,y):
    kmean = KMeans(n_clusters=4)
    kmean.fit(X)
    y_kmeans = kmean.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)
    centers =kmean.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()
def hiclus(X,y):
    ward2 = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
    ward3 = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
    ward4 = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward').fit(X)
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=ward2.labels_)
    plt.title("K=2")
    plt.subplot(132)
    plt.scatter(X[:, 0], X[:, 1], c=ward3.labels_)
    plt.title("K=3") 
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=ward4.labels_)
    plt.title("K=4")
    plt.show()
def svm(X,y):
    X, y = make_blobs(n_samples=50, centers=2,
                      random_state=0, cluster_std=0.60)
    plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap ='autumn')
    plt.show()
def randforest(X,y):
    forest = RandomForestClassifier(n_estimators = 100)
    forest.fit(X, y)
    forest.score(X,y)
    print("#Errors: %i" % np.sum(y != forest.predict(X)))
    plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap ='autumn')
    plt.show()
    
    

        
    
    
    
