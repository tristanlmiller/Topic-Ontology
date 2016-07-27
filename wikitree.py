# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:24:47 2016

@author: diyadas

This script plots trees of our wikipedia data.

"""

from ete3 import Tree, TreeStyle, NodeStyle

with open('/Users/diyadas/cdips/Topic-Ontology/SimpleWikiTree_u.txt','r') as f:
    treestr = f.readlines()[0]    
    

t = Tree( treestr.rstrip(),format=8)

circular_style = TreeStyle()
circular_style.mode = "c" # draw tree in circular mode
circular_style.scale = 120
circular_style.show_leaf_name = True
circular_style.show_branch_length = True
circular_style.show_branch_support = True
t.render("mytree.png", tree_style=circular_style)


nstyle = NodeStyle()
nstyle["hz_line_width"] = 3
nstyle["vt_line_width"] = 3

# Applies the same static style to all nodes in the tree. Note that,
# if "nstyle" is modified, changes will affect to all nodes
for n in t.traverse():
   n.set_style(nstyle)



ts = TreeStyle()
ts.branch_vertical_margin = 10
ts.show_leaf_name = True
ts.rotation = 90
ts.scale=100
t.render("tree_test100.png",tree_style=ts)

ts.scale=1000
t.render("tree_test1000.png",tree_style=ts)
## compare to

#t = Tree( '("[a,b]",c);' )
#t.show()

# []