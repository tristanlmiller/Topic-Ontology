'''
vec_to_cloud.py
Created by Diya Das on 28 Jul 2016

This script contains functions for generating word clouds.
'''

from wordcloud import WordCloud
import numpy as np
import pickle 
import itertools

def vec2cloud(pkltf = 'Tfidf_Matrix.pkl',labpkl='Feature_List.pkl' ):
    word_array = vec2words(pkltf, labpkl)
    
    

def vec2words(pkltf,labpkl ):
    tf = pickle.load(open(pkltf,'rb')).toarray()
    lab = pickle.load(open(labpkl,'rb'))
    freqInt = np.array(tf/ np.min(tf[np.nonzero(tf)]) , dtype='int')
    word_array = np.apply_along_axis(lambda b: ' '.join([ item for sublist in [[lab[i]]*b[i] for i in range(len(lab))] for item in sublist]),1,freqInt)
    return word_array
