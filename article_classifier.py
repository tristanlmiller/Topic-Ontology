'''
article_classifier
Author: Tristan Miller
Takes any string, finds the wikipedia article, and outputs the correct leaf
'''
import pandas as pd
import numpy as np
import nlp_api
import iterparsing_titles
import wikipedia as wiki
import string
import re
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import *

class classifier(object):
    def __init__(self,Tfidf=pd.read_pickle("Tfidf_weights.pkl"),
                 term_list=pd.read_pickle("Feature_List.pkl"),
                 PCA_matrix=pd.read_pickle("PCA_Components.pkl"),
                 c_means=pd.read_pickle("topic_means.pkl"),
                 descriptive_tree=pd.read_pickle("descriptive_tree.pkl"),
                 ward_tree=pd.read_pickle("ward_tree.pkl")):
        self.Tfidf = Tfidf
        self.term_list = term_list
        self.PCA_matrix = PCA_matrix
        self.c_means = c_means
        self.descriptive_tree = descriptive_tree
        self.ward_tree = ward_tree

    def classify_article(self,title):
        #Find the wikipedia article
        newtitles, text, links = iterparsing_titles.get_text([title],'')
        df = iterparsing_titles.get_dataframe(newtitles, text, links)

        if len(newtitles) == 0:
            print("Could not find unique article, or article too short.")
            return

        #apply nlp
        df['proctext'] = df['text'].apply(lambda x: nlp_api.para_to_stems(x) )
        df['proclinks'] = df['links'].apply(lambda x: ' '.join([item.lower().replace("-","_").replace(" ", "_").translate(str.maketrans({key: None for key in '().'})) for item in x]))

        #add links to text
        my_text = df['proctext'][0]
        my_text += (' ' + df['proclinks'][0])*2

        #vectorize
        vectorizer = CountVectorizer(analyzer = "word",vocabulary=self.term_list)
        article_vector = vectorizer.fit_transform([my_text])
        article_weighted = self.Tfidf.transform(article_vector).toarray()
        article_topic = np.dot(article_weighted,np.transpose(self.PCA_matrix))
        #now calculate the closest cluster mean
        min_distance = np.inf
        for i in range(self.c_means.shape[0]):
            current_distance = euclidean(article_topic,self.c_means[i,:])
            if current_distance < min_distance:
                min_distance = current_distance
                closest_cluster = i

        #Find the appropriate leaf in the descriptive_tree and return its name
        for d_node,l_node in zip(self.descriptive_tree.iter_leaves(),self.ward_tree.iter_leaves()):
            if closest_cluster == l_node.data:
                return d_node.data
        print("error, unexpected result")
        return
