# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:16:24 2016

@author: Tristan

This module applies ward clustering
Creates a tree of topics
Flattens the tree so that it's no longer binary
Creates another tree with descriptive names
"""

import pandas as pd
import numpy as np
from cluster_tree import *
import time
from sklearn.cluster import AgglomerativeClustering

def ward_cluster(file_feature_matrix="PCA_matrix.pkl",
                 file_link_matrix="Corrected_Link_Tfidf_Matrix.pkl",
                 file_link_list="Corrected_Link_Feature_List.pkl",n_clusters=100,
                 truncate=10000,output="",tolerance=0):
    #Only keep the first 10k articles by default.  I couldn't get it to run with 50k.
    #However, I got it with 25k, and it took 45 minutes.
    start_time = time.time()
    #Load pickles
    feature_matrix = pd.read_pickle(file_feature_matrix)[:truncate,:]
    link_matrix = pd.read_pickle(file_link_matrix)[:truncate,:]
    link_list = pd.read_pickle(file_link_list)
    processing_time = (time.time() - start_time)/60
    print("Current time: %.2f minutes.  Files loaded." % processing_time )

    #Apply clustering
    clustering = AgglomerativeClustering(linkage="ward", n_clusters=n_clusters)
    classification = clustering.fit_predict(feature_matrix)
    processing_time = (time.time() - start_time)/60
    print("Current time: %.2f minutes.  Clustering done." % processing_time )

    #Translate to tree_node, generate label_tree and collapsed_tree
    full_tree = tree_to_nodes(clustering.children_,feature_matrix.shape[0])
    label_tree = get_label_tree(classify_tree(full_tree,classification))
    (cluster_means,docs_in_cluster) = get_means(classification,feature_matrix)
    collapsed_tree = collapse_label_tree(label_tree,cluster_means,docs_in_cluster,tolerance)
    processing_time = (time.time() - start_time)/60
    print("Current time: %.2f minutes.  Trees collapsed." % processing_time )

    #Assign names to each node of tree based on most common links
    (link_means,docs_in_cluster) = get_means(classification,link_matrix)
    descriptive_tree = get_name_tree(collapsed_tree,link_means,docs_in_cluster,link_list)
    processing_time = (time.time() - start_time)/60
    print("Current time: %.2f minutes.  Node descriptions generated." % processing_time )

    #Write pickles
    #c_labels tells you which cluster each document is in
    pd.to_pickle(classification,output+'c_labels.pkl')
    #save the uncollapsed tree in case I want to tweak that process later.
    pd.to_pickle(label_tree,output+'uncollapsed_tree.pkl')
    #ward_tree shows how the clusters fit together in a tree structure
    pd.to_pickle(collapsed_tree,output+'ward_tree.pkl')
    #descriptive_tree the same as ward_tree, except that nodes are named after most common links
    pd.to_pickle(descriptive_tree,output+'descriptive_tree.pkl')
    #cluster_means are the vectors (in the PCA space) of each cluster
    pd.to_pickle(cluster_means,output+'cluster_means.pkl')
    #link_means are the vectors (in link space) of each cluster
    pd.to_pickle(link_means,output+'link_means.pkl')
    #docs_in_cluster tells you the size of each cluster
    pd.to_pickle(docs_in_cluster,output+'docs_in_cluster.pkl')
    processing_time = (time.time() - start_time)/60
    print("Current time: %.2f minutes.  Pickles written." % processing_time )
