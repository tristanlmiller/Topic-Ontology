''' 
Exploring Agglomerative Clustering
Author: Tristan Miller
I originally intended to make a validation measure for agglomerative clustering, but a lot of this is just exploring so far.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import *
import scipy
from sklearn.decomposition import PCA
#import lda #use pip install lda 
import time
from sklearn.cluster import AgglomerativeClustering
import sys
import re
import wikipedia #use pip install wikipedia
from scipy.spatial.distance import euclidean

#Load data, document-term matrix
data = pd.read_pickle("en_wikipedia_titles.pkl_df_500001_nlp.pkl")

#### vectorize text + links, giving extra weight to links
#unweighted_words = np.reshape(np.load("document_term_matrix.npy"),(1))[0]
#term_list = pd.read_pickle("term_list.pkl")[0].tolist()

#Cap the word frequency at 10.
#This is necessary at least for now with all those list articles
tens = np.zeros(unweighted_words.shape)+10
unweighted_words = scipy.sparse.csr_matrix(np.minimum(unweighted_words.toarray(),tens))

#Apply TF-IDF weighting
Tfidf = TfidfTransformer()
Tfidf.fit(unweighted_words)
weighted_words = Tfidf.transform(unweighted_words)

#Apply PCA with 30 components
start_time = time.time()
model = PCA(n_components=400)
doc_topic = model.fit_transform(weighted_words.toarray())
topic_word = model.components_
processing_time = (time.time() - start_time)/60
print("Time to execute: %.2f minutes" % processing_time )

plt.scatter(doc_topic[:,0], doc_topic[:,1])

#Ward Agglomerative clustering
start_time = time.time()
clustering = AgglomerativeClustering(linkage="ward", n_clusters=20)
c_labels = clustering.fit_predict(doc_topic)
processing_time = (time.time() - start_time)/60
print("Time to execute: %.2f minutes" % processing_time )


#Vectorize the links too
#links need to be put into the proper format
links = data["links"].copy()
for i in range(len(links)):
    for j in range(len(links[i])):
        links[i][j] = re.sub(' ','_',links[i][j])
for i in range(len(links)):
    links[i] = " ".join( links[i] )
    

vectorizer = CountVectorizer(analyzer = "word",min_df=.005)
link_vector = vectorizer.fit_transform(links)
link_names = vectorizer.get_feature_names()
link_vector.shape
plt.scatter(doc_topic[:, 0], doc_topic[:, 1], c=c_labels,
                        cmap=plt.cm.spectral)
                        
                        
if 'cluster_tree' in sys.modules:
    del sys.modules['cluster_tree'] #this just makes sure the import updates every time
from cluster_tree import *

root = tree_to_nodes(children,10000)

#Now I can classify each leaf
classified_tree = classify_tree(root,c_labels)
#And produce a reduced tree that only shows the clusters relative to each others
label_tree = get_label_tree(classified_tree)
str(label_tree)

#pip install ete3
#this is from here: http://etetoolkit.org/docs/latest/tutorial/tutorial_drawing.html
from ete3 import Tree, TreeStyle
t = Tree(str(label_tree))
ts = TreeStyle()
ts.show_leaf_name = True
ts.scale =  12
t.show(tree_style=ts)
#output is in a different window.

#Get all cluster means.
(cm,docs_in_cluster) = get_means(c_labels,doc_topic)
(link_means,docs_in_cluster) = get_means(c_labels,link_vector)

#now let's apply the algorithm described above, with zero tolerance
collapsed_tree = collapse_label_tree(label_tree,cm,docs_in_cluster,0)
str(collapsed_tree)

t = Tree(str(collapsed_tree))
t.show(tree_style=ts)

collapsed_tree = collapse_label_tree(label_tree,cm,docs_in_cluster,-.21)
str(collapsed_tree)

t = Tree(str(collapsed_tree))
t.show(tree_style=ts)

#pip install wordcloud
#see example here: https://github.com/amueller/word_cloud/blob/master/examples/simple.py
#Windows machines have trouble doing this installation, so maybe someone else should do this one.
from wordcloud import WordCloud
testcloud = WordCloud().generate('hello world')

def words_from_branch(c_mean,term_list,topics=None):
    #If topic_word is given, then it's assumed that c_mean is in topic space
    #and topic_word is used to translate topic space to word space.
    if(topics is not None):
        bag = np.dot(c_mean,topics)
    else:
        bag = c_mean
    df = pd.DataFrame(index=range(len(bag)),columns=['Term','Weight'])
    df['Term'] = term_list
    df['Weight'] = bag
    df.sort_values('Weight',ascending=False,inplace=True)
    return df

c_mean,documents = get_branch_mean(collapsed_tree[2][3],cm,docs_in_cluster)
words_from_branch(c_mean,term_list,topics=topic_word).head()

#I can also find the most common links in a cluster
l_mean,documents = get_branch_mean(collapsed_tree[2][3],link_means,docs_in_cluster)
words_from_branch(l_mean,link_names).head()


#To classify an article, we need 5 things.
#The article text
#A list of the terms for vectorization
#The TF-IDF transformation
#The topic_word matrix from PCA analysis
#The means of the clusters in topic space
def classify_article(article_text,term_list,Tfidf,topic_word,c_means):
    vectorizer = CountVectorizer(analyzer = "word",vocabulary=term_list)
    article_vector = vectorizer.fit_transform([article_text])
    article_weighted = Tfidf.transform(article_vector).toarray()
    article_topic = np.dot(article_weighted,np.transpose(topic_word))
    #now calculate the closest cluster mean
    min_distance = np.inf
    for i in range(c_means.shape[0]):
        current_distance = euclidean(article_topic,c_means[i,:])
        if current_distance < min_distance:
            min_distance = current_distance
            closest_cluster = i
    
    return closest_cluster
    
cluster_id = classify_article(wikipedia.summary('One Direction'),term_list,Tfidf,topic_word,cm)
words_from_branch(cm[cluster_id,:],term_list,topics=topic_word).head()

