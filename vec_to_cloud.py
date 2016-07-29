'''
vec_to_cloud.py
Created by Diya Das on 28 Jul 2016

This script contains functions for generating word clouds.
'''

from wordcloud import WordCloud
import numpy as np
import pickle 
import cluster_tree
import matplotlib.pyplot as plt

def vec2cloud(pkltf = 'Tfidf_Matrix.pkl',cmeanspkl = "cluster_means.pkl", docpkl = "docs_in_cluster.pkl", wardtreepkl = "ward_tree.pkl", labpkl='Feature_List.pkl' ):
    word_array = vec2words(pkltf, labpkl, cmeanspkl, docpkl, wardtreepkl)
    word_array = word_array[0:20] #for testing
    for i in range(len(word_array)):
        wc = WordCloud(background_color="black")
        wc = wc.generate(word_array[i])
        wc.to_file("cloud"+str(i)+".png")
        del wc #this was giving me issues earlier, so hopefully this fixes it

def vec2words(pkltf,labpkl , cmeanspkl, docpkl, wardtreepkl):
    tree= pickle.load(open(wardtreepkl,"rb"))
    lab = pickle.load(open(labpkl,'rb'))
    tf = np.empty((len(tree.iter_nodes()),len(lab)))
    clabels = pickle.load(open("c_labels.pkl",'rb'))
    tf_idf = pickle.load(open(pkltf,'rb')).toarray()
    counter=0;
    clustermeans,docsincluster = cluster_tree.get_means(clabels,tf_idf[:25000,:])
    for node in tree.iter_nodes():
        tf[counter,:],ndocs = cluster_tree.get_branch_mean(node, clustermeans, docsincluster)
        counter += 1
    freqInt = np.array(tf/ np.min(tf[np.nonzero(tf)]) , dtype='int')
    word_array = np.apply_along_axis(lambda b: ' '.join([ item for sublist in [[lab[i]]*b[i] for i in range(len(lab))] for item in sublist]),1,freqInt)
    return word_array
