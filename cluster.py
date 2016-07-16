# -*- coding: utf-8 -*-
"""
Created on Sat 16 Jul 2016

@author: Diya

snippets of clustering code

"""

from sklearn.cluster import AgglomerativeClustering

def wardclus(X,X_red, n)
    clustering = AgglomerativeClustering(linkage="ward", n_clusters=n)
    clustering.fit(X_red)
    plot_clustering(X_red, X, clustering.labels_, "Ward linkage")
    plt.show()

    #X is PCA transformed data, X_red is 2D representation, n is number of clusters

