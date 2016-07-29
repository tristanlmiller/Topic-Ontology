# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:02:10 2016

@author: Tristan

I'm writing functions related to the tree output of agglomerative clustering
"""
import numpy as np
from scipy.spatial.distance import euclidean

#sklearn's agglomerative clustering returns trees in a weird format
#here I convert to the more intuitive format, where each node is a list of its children
def tree_to_list(children,num_docs):
    node_list = [[]]*len(children)
    finished = False
    loop_count = 0
    while((not finished) and loop_count < num_docs):
        #Loop through children multiple times, until no more nodes are added
        finished = True
        loop_count += 1 #just a safeguard to prevent infinite loops
        for i in range(len(children)):
            #For each node, check it's been added, and if not, whether it could be added
            if node_list[i] == []:
                #A string can be added if each child is either a document, or a node with a known string
                flag = True
                current_list = []
                for j in range(len(children[i])):
                    if (children[i][j] < num_docs):
                        current_list.append(children[i][j])
                    elif children[i][j] >= num_docs and node_list[children[i][j]-num_docs] != []:
                        current_list.append(node_list[children[i][j]-num_docs])
                    else:
                        flag = False
                if(flag):
                    node_list[i] = current_list.copy()
                    last_node = i
                    finished = False
    
    #finally, return root
    return node_list[last_node]
    
#I'm also trying programming my own class.  Might be better, might be worse?
class TreeNode(object):
    def __init__(self,data):
        self.data = data
        self.children = []
        #All the children should be TreeNodes
    
    def append(self,obj):
        self.children.append(obj)
    
    def __getitem__(self,key):
        return self.children[key]
    
    def __setitem__(self,key,value):
        self.children[key] = value
        
    def __delitem__(self,key):
        del self.children[key]
        
    def __len__(self):
        return len(self.children)
    
    #Return a list of all nodes in this tree
    def iter_nodes(self):
        myIter = [self]
        for i in range(len(self.children)):
            myIter.extend(self.children[i].iter_nodes())
        return myIter
    
    #Return a list of all leaves in this tree
    def iter_leaves(self):
        myIter = []
        if len(self.children) == 0:
            myIter.append(self)
        else:
            for i in range(len(self.children)):
                myIter.extend(self.children[i].iter_leaves())
        return myIter
    
    #Return a separate copy of this tree
    def copy(self):
        myCopy = TreeNode(self.data)
        for child in self.children:
            myCopy.append(child.copy())
        return myCopy
    
    #str(TreeNode) also returns exactly what you want for the ete package
    def __str__(self):
        if(len(self.children) > 0):
            tree_string = "("
            for child in self.children:
                tree_string += str(child)[:-1] + ','
            tree_string = tree_string[:-1] + ')'
        else:
            tree_string = str(self.data)
        tree_string += ';'
        return tree_string
    
    #Join this node with one of its children.
    #The data in the child node will be used to overwrite this one.
    def join_child(self,child_index):
        joined_child = self[child_index]
        del self[child_index]
        for child in joined_child.children:
            self.append(child)
        self.data = joined_child.data
        
#Translate instead to my TreeNode object
def tree_to_nodes(children,num_docs):
    node_list = [None]*len(children)
    finished = False
    loop_count = 0
    while((not finished) and loop_count < num_docs):
        #Loop through children multiple times, until no more nodes are added
        finished = True
        loop_count += 1 #just a safeguard to prevent infinite loops
        for i in range(len(children)):
            #For each node, check it's been added, and if not, whether it could be added
            if node_list[i] is None:
                #A string can be added if each child is either a document, or a node with a known string
                flag = True
                current_list = TreeNode(None)
                for j in range(len(children[i])):
                    if (children[i][j] < num_docs):
                        current_list.append(TreeNode(children[i][j]))
                    elif (children[i][j] >= num_docs) and (node_list[children[i][j]-num_docs] != None):
                        current_list.append(node_list[children[i][j]-num_docs])
                    else:
                        flag = False
                if(flag):
                    node_list[i] = current_list
                    last_node = i
                    finished = False
    
    #finally, return root
    return node_list[last_node]

#Given a TreeNode whose leaf data are document numbers, and a list of labels for the documents
#this returns a tree whose data are the given labels
def classify_tree(root,labels):
    classified_tree = root.copy()
    for node in classified_tree.iter_leaves():
        node.data = labels[node.data]
    return classified_tree

#Given a TreeNode with labeled leaves, collapses it until no redundant branches remain
#branches are redundant if all the children of a node are labeled the same way
def get_label_tree(root):
    label_tree = root.copy()
    for node in label_tree.iter_nodes()[::-1]:
        #consider the nodes in reverse order, which guarantees that children
        #will always come before their parents.
        if(len(node) > 0):
            #Check whether all children have the same labels
            my_label = node[0].data
            all_same = my_label is not None
            for child in node.children:
                if child.data != my_label:
                    all_same = False
            
            if all_same:
                #If they do all have the same labels, then remove the children
                node.children = []
                node.data = my_label
    return label_tree

#Returns an ndarray containing all the cluster means
#Also returns the number of docs within each cluster
def get_means(c_labels, feature_matrix):
    num_clusters = max(c_labels)+1
    num_terms = feature_matrix.shape[1]
    num_docs = feature_matrix.shape[0]
    
    #each row of cm corresponds to a cluster
    #and each column a term
    cm=np.zeros((num_clusters,num_terms))
    docs_in_cluster = np.zeros((num_clusters))
    
    for doc_index in range(num_docs):
        cm[c_labels[doc_index],:]+=feature_matrix[doc_index,:]
        docs_in_cluster[c_labels[doc_index]] += 1
    for cluster_index in range(num_clusters):
        cm[cluster_index,:] /= docs_in_cluster[cluster_index]
    return cm, docs_in_cluster

#Returns the euclidean distance between two cluster means
def get_dist(cluster_means,c1,c2):
    return euclidean(cluster_means[c1,:],cluster_means[c2,:])

#returns the mean of a branch, along with the number of documents within
#The branch is understood to be a branch in the label_tree
def get_branch_mean(branch,cm,docs_in_cluster):
    if len(branch) == 0:
        #If this is a leaf, the mean can just be looked up in the table.
        return cm[branch.data,:], docs_in_cluster[branch.data]
    else:
        #If this is not a leaf, take a weighted average of each of its children
        total_docs = 0
        my_mean = np.zeros(cm.shape[1])
        for child in branch.children:
            child_mean, child_docs = get_branch_mean(child,cm,docs_in_cluster)
            my_mean += child_mean*child_docs
            total_docs += child_docs
        my_mean /= total_docs
        return my_mean, total_docs

#Given two branches in the label tree, determines the distance between their means
def branch_mean_distance(branch1,branch2,cm,docs_in_cluster):
    #First find the mean of each branch
    mean1,docs1 = get_branch_mean(branch1,cm,docs_in_cluster)
    mean2,docs2 = get_branch_mean(branch2,cm,docs_in_cluster)
    return euclidean(mean1,mean2)
    
#This goes through a label tree, and collapses nodes according to the following rule:
#A node is collapsed with a non-leaf child if its grandchildren are no closer to each other
#than its children are to the grandchildren.
def collapse_label_tree(label_tree,cm,docs_in_cluster,tolerance):
    collapsed_tree = label_tree.copy()
    #Iterate through the nodes backwards
    #so that children are always considered before parents
    for parent in collapsed_tree.iter_nodes()[::-1]:
        #iterate through children, noting that the children may change throughout the loop
        child_index = 0
        while(child_index < len(parent)):
            child = parent[child_index]
            #only consider non-leaf children
            if(len(child) > 0):
                #compute grandchild_separation, the min distance among the child's branches
                #In principle, there may be more grandchildren.  Check each pair
                grandchild_separation = np.inf
                for grandchild1 in child.children:
                    for grandchild2 in child.children:
                        if grandchild2 != grandchild1:
                            grandchild_separation = min([branch_mean_distance(grandchild2,grandchild1,cm,docs_in_cluster),grandchild_separation])
                    
                #now, compute child_separation, the min distance between a single child
                #and a single grand child
                child_separation = np.inf
                for child2 in parent.children:
                    for grandchild in child.children:
                        if child2 != child:
                            child_separation = min([branch_mean_distance(child2,grandchild,cm,docs_in_cluster),child_separation])
                
                #now we compare child_separation and grandchild_separation
                if grandchild_separation*(1+tolerance) > child_separation:
                    #delete the child from the parent's list of children
                    del parent[child_index]
                    #add the grandchildren to the list of children,
                    #inserting them in the right location
                    for grandchild in child.children:
                        parent.children.insert(child_index,grandchild)
                else:
                    #move on to the next child
                    child_index += 1
            else:
                #If this child is a leaf, move on.
                child_index += 1
                    
    return collapsed_tree
   
#This function takes a label tree, and assigns each node an informative name
def get_name_tree(label_tree,cm,docs_in_cluster,term_list):
    name_tree = label_tree.copy()
    #Iterate through each non-leaf, going from the top of the tree to the bottom.
    for parent in name_tree.iter_nodes():
        if(len(parent) > 0):
            #Now I want to go through each child and figure out what distinguishes that child from the others
            parent_mean,docs = get_branch_mean(parent,cm,docs_in_cluster)
            for child in parent.children:
                child_mean,docs = get_branch_mean(child,cm,docs_in_cluster)
                #Just subtract the parent mean from the child mean
                child_mean -= parent_mean
                
                df = pd.DataFrame(index=range(len(child_mean)),columns=['term','frequency'])
                df['term'] = term_list
                df['frequency'] = child_mean
                df.sort_values('frequency',inplace=True,ascending=False)
                child.data = df['term'].tolist()[0] + '/' + df['term'].tolist()[1] + '/' + df['term'].tolist()[2]
    
    return name_tree
    
    
    