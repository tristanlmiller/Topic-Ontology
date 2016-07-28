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

def vec2cloud(pkltf = 'Tfidf_Matrix.pkl',labpkl='Feature_List.pkl' ):
    word_array = vec2words(pkltf, labpkl)
    word_array = word_array[0:20] #for testing
    for i in range(len(word_array)):
        wc = WordCloud(background_color="black")
        wc = wc.generate(word_array[i])
        wc.to_file("cloud"+str(i)+".png")
        del wc #this was giving me issues earlier, so hopefully this fixes it


def vec2words(pkltf,labpkl ):
    tf = pickle.load(open(pkltf,'rb')).toarray()
    lab = pickle.load(open(labpkl,'rb'))
    freqInt = np.array(tf/ np.min(tf[np.nonzero(tf)]) , dtype='int')
    word_array = np.apply_along_axis(lambda b: ' '.join([ item for sublist in [[lab[i]]*b[i] for i in range(len(lab))] for item in sublist]),1,freqInt)
    return word_array
