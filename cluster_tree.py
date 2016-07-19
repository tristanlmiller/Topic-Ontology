# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:02:10 2016

@author: Tristan

I'm writing functions related to the tree output of agglomerative clustering
"""

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

    
    