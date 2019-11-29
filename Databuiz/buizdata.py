
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
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
def train(X,y):
    X_train,X_test,y_train,y_test =train_test_split(X,y,test_size =0.3,random_state =0) 
    lrl1 =linear_model.LogisticRegression(penalty ='l1')
    lrl1.fit(X,y)
    print ("Train - Accuracy :", metrics.accuracy_score(y_train, lrl1.predict(X_train)))
    print ("Train - Confusion matrix:",metrics.confusion_matrix(y_train,lrl1.predict(X_train)))
    print ("Test - Accuracy :", metrics.accuracy_score(y_test, lrl1.predict(X_test)))
    print ("Test - Confusion matrix:",metrics.confusion_matrix(y_test,lrl1.predict(X_test)))
def lda(X,y):
    lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))
    lda =LDA()
    proj =lda.fit(X,y)
    proj1 =lda.transform(X)
    y_pred_lda =lda.predict(X)
    errors  =y_pred_lda != y
    print(errors)
    print("R-squared =", metrics.r2_score(y, y_pred_lda))
    plt.scatter(X[:,0],X[:,1])
    plt.plot(y)
    plt.show()
def logReg(X,y):
    lrl1 =linear_model.LogisticRegression(penalty ='l1')
    lrl1.fit(X,y)
    y_pred_lrl1 = lrl1.predict(X)
    errors = y_pred_lrl1 != y
    print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_lrl1)))
    print(lrl1.coef_)
    print("R-squared =", metrics.r2_score(y, y_pred_lrl1))
    plt.scatter(X[:,0],X[:,1])
    plt.plot(y)
    plt.show()
def pca(X,y): 
    pca =PCA(n_components =2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    PC =pca.transform(X)
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.subplot(122)
    plt.scatter(PC[:, 0], PC[:, 1])
    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
                



	
    
    
          
        
    
  
