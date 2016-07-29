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

    def __init__(self, dataframe, num_clusters, feature_matrix, min_cluster_size, max_depth):
        self.dataframe = dataframe
        self.num_clusters = num_clusters
        self.feature_matrix=feature_matrix
        self.n_samples=dataframe.shape[0]
        self.min_cluster_size=min_cluster_size
        self.max_depth=max_depth

    def get_cluster_labels(self, feature_matrix, num_clusters): #returns vector of labels
        kmeans=KMeans(n_clusters=num_clusters)
        labels=kmeans.fit_predict(feature_matrix)
        return labels

    def remove_rows(self, sparse_matrix, index): #if index=[2,4,6] removes rows 2 4 6 from matrix
        return numpy.delete(sparse_matrix, index, 0)

    def get_size(self, cluster_label, level, df):
        count=0
        for x in df['Level_'+str(level)]:
            if x==cluster_label:
                count+=1
        return count


    @staticmethod
    def tuple_append(t, entry): #add entry to end of tuple t, like list append
        x=list(t)
        x.append(entry)
        return(tuple(x))

    def get_final_labels(self, df): #returns vector of final cluster labels, after Level data has been added in Run()
        labels=[]
        for i in df.index.values:
            j=1
            while df['Level_'+str(j)][i]==None:
                j+=1
            labels.append(df['Level_'+str(j)][i])
        return labels

    def get_cluster_titles(self, cluster_label, final_labels):  # return list of all elements whose cluster label starts with cluster_label
        titles=[]
        cluster_depth=len(cluster_label)
        for i in range(self.dataframe.shape[0]):
            if final_labels[i][0:cluster_depth]==cluster_label: #check if beginning of final label matches given label
                titles.append(self.dataframe['title'][i])
        return titles

    def get_cm(self, label, final_labels): #return cm of all points whose label starts with label
        cm=[0]*self.feature_matrix.shape[1]
        count=0
        for i in range(len(final_labels)):
            if final_labels[i][0:len(label)]==label:
                cm+=self.feature_matrix[i, :]
                count+=1
        if count==0:
            print(str(label)+' empty cluster')
            return 0
        else:
            return cm/count

    def get_all_labels(self, final_labels): #get set of all labels including internal nodes
        labels=set(final_labels)
        for node in final_labels:
            for i in range(len(node)):
                labels=labels.union({node[0:i]})
        return labels

    def get_cm_dict(self, all_labels, final_labels): #returns dictionary of cm of each cluster in feature space, including internal nodes
        cm={}
        t0=time.time()
        for label in set(all_labels): #iterate over distinct labels
            cm[label]=self.get_cm(label, final_labels)
        print(time.time()-t0)
        return cm

    def get_cm_dict_word_space(self, cm_dict, pca_components): #multiply by tfidf matrix; cm_dict is dict of pca cms
        cm_dict_word_space={}
        for label in cm_dict.keys():
            cm_dict_word_space[label]=numpy.dot(cm_dict[label], pca_components)
        return cm_dict_word_space


    def get_nearest_cm(self,feature_vector, cm_dict): #given vector not in data set, finds closest cluster center
        min_dist=float('inf')
        for label in cm_dict.keys():
            if scipy.spatial.distance.euclidean(feature_vector, cm_dict[label])<min_dist:
                min_dist=scipy.spatial.distance.euclidean(feature_vector, cm_dict[label])
                min_label=label
        return min_label

    '''
    def get_distinct_labels(self, level): #not actually necessary for this implementation
        distinct_labels_as_tuples=list(set([tuple(x) for x in df['Level_'+str(level)]])) #set accepts tuples but not lists
        return [list(x) for x in distinct_labels_as_tuples]
    '''

    def Top_Words_in_Cluster(self, c_mean,term_list, number, topics=None): #return top (number) words in cluster
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
        df.index=range(df.shape[0])
        x=''
        for i in range(number):
            x+=df['Term'][i]
            x+='/'
        return x

    def top_words_in_cluster_dict(self, wordspace_cm_dict, term_list, number): #get dict of cm in wordspace of all cm, incl internal nodes
        d={}
        for label in wordspace_cm_dict.keys():
            d[label]=self.Top_Words_in_Cluster(wordspace_cm_dict[label], term_list, number)
        return d

    def get_topword_dict(self, pca_components, term_list, final_labels, number): #return dict of cm of all cluster, including internal nodes; this is what should be called from outside the class
        return self.top_words_in_cluster_dict(self.get_cm_dict_word_space(self.get_cm_dict(self.get_all_labels(final_labels), final_labels), pca_components), term_list, number)

    def get_sizes(self, final_labels): #return dict of sizes
        d={}
        for label in set(final_labels):
            d[label]=0
        for label in final_labels:
            d[label]+=1
        return d



    def Run(self): #returns vector of cluster lables; note each tuple begins with 0
        level=0
        df=self.dataframe
        #df['Final_Address']=[0]*self.n_samples
        df['Level_0']=[(0,)]*self.n_samples #need comma for single element tuple
        big_clusters=[(0,)]
        index=df.index.values
        while len(big_clusters)>0 and level<self.max_depth:
            t_0=time.time()
            sizes={} #records size of each newly created cluster
            level+=1
            df['Level_'+str(level)]=df['Level_'+str(level-1)] #create new column for next level
            for label in big_clusters:
                cluster_df=df[df['Level_'+str(level)]==label] #get rows in current cluster
                cluster_df['new_label']=[[]]*cluster_df.shape[0]
                cluster_index=cluster_df.index.values
                cluster_fm=self.remove_rows(self.feature_matrix, list(set(index).difference(set(cluster_index)))) #keep only rows from feature matrix corresponding to these articles
                labels=self.get_cluster_labels(cluster_fm, self.num_clusters) #perform clustering
                #print('label', labels[0])
                cluster_df['new_label']=labels
                count=[0]*self.num_clusters
                for i in cluster_index:
                    #print(type(df['Level_'+str(level)][i]), type(cluster_df['new_label'][i]))
                    a=df['Level_'+str(level)][i]
                    b=cluster_df['new_label'][i]


                    df['Level_'+str(level)][i]=self.tuple_append(a,b) #go to entry in original df and update
                    count[cluster_df['new_label'][i]]+=1 #compute size of each new cluster

                for i in range(self.num_clusters):
                    sizes[self.tuple_append(label, i)]=count[i] #record size, using total address as key

            for i in range(self.n_samples):
                if df['Level_'+str(level)][i]==df['Level_'+str(level-1)][i] and df['Level_'+str(level-1)][i]!=None:
                    #df['Final_Address'][i]=df['Level_'+str(level)][i]
                    df['Level_'+str(level)][i]=None #delete duplicate labels for clusters that weren't further divided
                else:
                    df['Level_'+str(level-1)][i]=None #keep only most recent label for clusters that were further divided


            big_clusters=[label for label in list(sizes.keys()) if sizes[label]>=self.min_cluster_size] #update with new clusters
            print(level,time.time()-t_0)
        return self.get_final_labels(df)

class Cluster_Tree(TreeNode, Iterative_KMeans): #this class reads the vector of cluster labels created by Iterative_KMeans.Run() and creates a tree
    def __init__(self, label_vector):
        self.nodes=set(label_vector)#the label vector might not contain all internal nodes, since it only records the *final* cluster label of each article
        for node in self.nodes: #add internal nodes not in label_vector
            for i in range(len(node)):
                self.nodes=self.nodes.union({node[0:i]})

    def get_root(self):
        return (0,)

    def get_children(self,node):
        children=set()
        for n in self.nodes:
            if n[0:len(node)]==node and len(n)==len(node)+1: #children of (3,5) are all nodes of form (3,5,x)
                children=children.union({n})
        return children

    def get_all_children(self, node): #return children, and childrenof children, and...
        children=set()
        for n in self.nodes:
            if n[0:len(node)]==node:
               children=children.union({n})
        return children

    def get_leaves(self):
        leaves=set()
        for n in self.nodes:
            if self.get_children(n)==set():
                leaves=leaves.union({n})
        return leaves

    def get_newick(self, root, names): #return representation as string of nested paranetheses; this is the format most tree visualizers want
        children=self.get_children(root)
        if children==set():
            newick=names[root]
        else:
            newick='('
            for child in children:
                x=self.get_newick(child, names)+','
                newick+=x

            y=newick[0:len(newick)-1] #delete last character, which is a comma
            newick=y+')'
            newick+=names[root]
        return newick

    def get_Root_as_TreeNode(self, cm_dict_word_space, labels): #return root as TreeNode class
        Root=TreeNode((0,))
        current_node=(0,)
        current_TN_node=Root
        i=1
        while i<max_depth+1:
            for m in Root.iter_leaves():
                for n in self.nodes:
                    if m.data==n[0:len(m.data)] and len(m.data)==len(n)-1:
                        m.append(TreeNode(n))
            i+=1
        return Root
