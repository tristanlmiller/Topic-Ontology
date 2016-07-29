"""
Created on Fri Jul 15
@author: Simon

Functions to compute basic statistics of the clusters.

feature_matrix is a sparse matrix obtained by vectorizing the text somehow, e.g. a Tf-idf or Count-frequency matrix.
This is the matrix used to perform the clustering.

list_of_clusters is an array of the clusters, where each cluster is represented as an array of titles.
e.g. list_of_clusters[0]=['France', 'England', 'Germany', ...]


The last 4 statistics are defined in:
http://uni-obuda.hu/conferences/mtn2005/KovacsFerenc.pdf
"""


import pickle
import scipy
import numpy
import pandas

num_articles=10000

with open('cleaned-10k-articles.pkl', 'rb') as pickle_file:
  data = pickle.load(pickle_file)


def get_lengths(list_of_clusters): #return array of cluster lengths
  lengths=[]
  for cluster in list_of_clusters:
  lengths.append(len(cluster))
  return lengths


def index_of_title(article_name):
  return data[data['title']==article_name].index.values[0]


def get_cm(cluster, feature_matrix): #return cm of cluster in feature space
  cm=[0]*len(feature_matrix[0, :].toarray())
  for title in cluster:
    cm+=feature_matrix[index_of_title(title), :].toarray()
  cm/=len(cluster)
  return(cm)


def get_variance(cluster, feature_matrix):
  var=0
  cm=get_cm(cluster)
  for title in cluster:
      var+= scipy.spatial.distance.euclidean(feature_matrix[index_of_title(title), :].toarray(), cm)**2
  return var/len(cluster)


def get_scattering(array_of_clusters, feature_matrix):
  scatt=0
  for cluster in array_of_clusters:
      scatt+=get_variance(cluster, feature_matrix)
  scatt/=(len(array_of_clusters)*get_variance(data['title'], feature_matrix))
  return scatt



def get_Dis(array_of_clusters, feature_matrix):
  cm_array=[]
  n_c=len(array_of_clusters)
  distance_matrix=[[0 for j in range(n_c)] for i in range(n_c)]
  for cluster in array_of_clusters:
      cm_array.append(get_cm(cluster, feature_matrix))
  for i in range(n_c):
      for j in range(n_c):
          distance_matrix[i][j]=scipy.spatial.distance.euclidean(cm_array[i], cm_array[j])
  Dis=0
  for i in range(n_c):
    inv_sum=0
    for j in range(n_c):
      if j!=i:
        inv_sum+=distance_matrix[i][j]
    Dis+=(inv_sum)**-1
  max_dist=numpy.asarray(distance_matrix).max(1).max(0)
  for i in range(n_c):
    for j in range(n_c):
      if distance_matrix[i][j]==0:
        distance_matrix[i][j]=float('inf')

  min_dist=numpy.asarray(distance_matrix).min(1).min(0)
  return max_dist*Dis/min_dist


def get_density(point, list_of_clusters, feature_matrix):
  count=0
  n_c=len(list_of_clusters)
  stdev=numpy.sqrt(sum([get_variance(cluster, feature_matrix) for cluster in list_of_clusters]))/n_c
  for x in range(num_articles):
      count+=(0<scipy.spatial.distance.euclidean(feature_matrix[x, :].toarray(), point)<stdev)
  return count


def get_Dens_bw(list_of_clusters): #This one takes a really long time to run!
  n_c=len(list_of_clusters)
  cm_array=[get_cm(cluster) for cluster in list_of_clusters]
  cm_density=[get_density(cm, aff_prop_clusters) for cm in cm_array]
  return sum([sum([get_density(.5*(cm_i+cm_j), aff_prop_clusters)/max(cm_density[i], cm_density[j]) for j in range(n_c)if j!=i]) for i in range(n_c)])/(n_c*(n_c-1))
