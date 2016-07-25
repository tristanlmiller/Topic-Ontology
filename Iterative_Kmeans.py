'''
Iterative kmeans

Author: Simon

Recursively apply K-Means to all clusters above desired size threshold.
The final cluster assignment is recorded as a tuple: e.g. (0,5,3) means
the 3rd cluster of the 5th cluster of the 0th cluster.

By convention, all tuples begin with 0; i.e. the 0th cluster is the entire dataset.

The set of clusters can also be thought of as a tree: the root is the tuple (0),
the nodes are the various tuples (i.e. clusters), and the children of a tuple X are the tuples of the form (X, l)
where l is a number. In other words, the children of a cluster are its subclusters.
'''
import pandas as pd
import numpy, scipy
from sklearn.cluster import KMeans

class Iterative_KMeans:

    def __init__(self, dataframe, num_clusters, feature_matrix, min_cluster_size): #note feature_matrix and dataframe must have same num of rows; also f_m should be sparse
        self.dataframe = dataframe
        self.num_clusters = num_clusters
        self.feature_matrix=feature_matrix
        self.n_samples=dataframe.shape[0]
        self.min_cluster_size=min_cluster_size

    def get_cluster_labels(self, feature_matrix, num_clusters): #performs kmeans and returns vector of labels
        kmeans=KMeans(n_clusters=num_clusters)
        labels=kmeans.fit_predict(feature_matrix)
        return labels

    def remove_rows(self, sparse_matrix, index): #if index=[2,4,6] removes rows 2 4 6 from matrix
        return numpy.delete(sparse_matrix.toarray(), index, 0)


    @staticmethod
    def tuple_append(t, entry): #add entry to end of tuple t, like list append
        x=list(t)
        x.append(entry)
        return(tuple(x))

    def get_final_labels(self, df): #returns vector of final cluster labels, after Level data has been added in Run(); labels is a vector of tuples
        labels=[]
        for i in df.index.values:
            j=1
            while df['Level_'+str(j)][i]==None:
                j+=1
            labels.append(df['Level_'+str(j)][i])
        return labels

    def get_cluster_titles(self, cluster_label, final_labels):  # return list of all elements whose cluster label starts with cluster_label; note c_l is a tuple and f_l is vector of tuples
        titles=[]
        cluster_depth=len(cluster_label)
        for i in range(self.dataframe.shape[0]):
            if final_labels[i][0:cluster_depth]==cluster_label: #check if beginning of final label matches given label
                titles.append(self.dataframe['title'][i])
        return titles

    def get_cm(self, label, final_labels): #return cm in feature space of all points in cluter labeled by label
        cm=[0]*shape.self.feature_matrix[1]
        count=0
        for i in range(len(final_labels)):
            if final_labels[i]==label:
                cm+=self.feature_matrix(i, :).toarray() #if matrix is sparse
                count+=1
        if count==0:
            print('Warning: Cluster '+str(label)+' is empty')
            return 0
        else:
        return cm/count


    def get_cm_dict(self, final_labels): #returns a dictionary with entries of the form {(3,4): cm of cluster (3,4)}
        cm={}
        for label in set(final_labels): #iterate over distinct labels
            cm[label]=self.get_cm(label, final_labels)
        return cm

    def get_nearest_cm(self,feature_vector, cm_dict): #given feature vector not in data set, finds closest cluster center
        min_dist=float('inf')
        for label in cm_dict.keys:
            if scipy.spatial.distance.euclidean(feature_vector, cm_dict[label])<min_dist:
                min_dist=scipy.spatial.distance(feature_vector, cm_dict[label])
                min_label=label
        return min_label

        '''
        these are not strictly necessary for this implementation, but may be useful in some situations

    def get_size(self, cluster_label, level, df):
        count=0
        for x in df['Level_'+str(level)]:
            if x==cluster_label:
                count+=1
        return count


        def get_distinct_labels(self, level): #not actually necessary for this implementation
            distinct_labels_as_tuples=list(set([tuple(x) for x in df['Level_'+str(level)]])) #set accepts tuples but not lists
            return [list(x) for x in distinct_labels_as_tuples]

        '''

    def Run(self): #returns vector of cluster lables; each entry is a tuple; note each tuple begins with 0
        level=0
        df=self.dataframe
        df['Level_0']=[(0,)]*self.n_samples #need comma for single element tuple
        big_clusters=[(0,)]
        index=df.index.values
        while len(big_clusters)>0:
            sizes={} #records size of each newly created cluster
            level+=1
            df['Level_'+str(level)]=df['Level_'+str(level-1)] #create new column for next level
            for label in big_clusters: #loop though cluster with >=min_cluster_size elements
                cluster_df=df[df['Level_'+str(level)]==label] #get rows in current cluster
                cluster_df['new_label']=[[]]*cluster_df.shape[0]
                cluster_index=cluster_df.index.values
                cluster_fm=self.remove_rows(self.feature_matrix, list(set(index).difference(set(cluster_index)))) #keep only rows from feature matrix corresponding to these articles
                labels=self.get_cluster_labels(cluster_fm, self.num_clusters) #perform clustering
                cluster_df['new_label']=labels
                count=[0]*self.num_clusters #initialize array to store length of each new cluster
                for i in cluster_index: #loop through indices of original cluster
                    a=df['Level_'+str(level)][i]
                    b=cluster_df['new_label'][i] #for ease of reading
                    df['Level_'+str(level)][i]=self.tuple_append(a,b) #go to entry in original df and update
                    count[cluster_df['new_label'][i]]+=1 #compute size of each new cluster

                for i in range(self.num_clusters): #loop thorugh new clusters
                    sizes[self.tuple_append(label, i)]=count[i] #record size, using total address as key
                    #sizes records size of all new clusters, not just those created from a given label; the above step ensures sizes has separate entries e..g (0,1,2) and (0,2,2) (here label is the first two elements)
            for i in range(self.n_samples): #loop through dataframe and remove redundant entries
                if df['Level_'+str(level)][i]==df['Level_'+str(level-1)][i] and df['Level_'+str(level-1)][i]!=None:
                    df['Level_'+str(level)][i]=None #delete duplicate labels for clusters that weren't further divided
                else:
                    df['Level_'+str(level-1)][i]=None #keep only most recent label for clusters that were further divided


            big_clusters=[label for label in list(sizes.keys()) if sizes[label]>=self.min_cluster_size] #update with new clusters
        return self.get_final_labels(df) #return vector of tuples


class Cluster_Tree: #this class reads the vector of cluster labels created by Iterative_KMeans.Run() and creates a tree
    def __init__(self, label_vector): #label vector should be a list of tuples, like the output of Iterative_KMeans.Run()
        self.nodes=set(label_vector) #converting to a set removes redundant labels

    def get_children(self,node): #return direct children of a node
        children=[]
        for n in self.nodes:
            if len(n)==len(node)+1 and n[0:len(node)]==node: #children of node (3,5) are all tuples n=(3,5,x)
                children.append(n)
        return children

    def get_all_children(self, node): #return all children, and all children of children, and...
        children=[]
        for n in self.nodes:
            if n[0:len(node)]==node:
                children.append(n)
        return children
