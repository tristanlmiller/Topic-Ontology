# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:24:47 2016

@author: diyadas

This script plots trees of our wikipedia data.

"""

from ete3 import Tree

with open('/Users/diyadas/cdips/Topic-Ontology/SimpleWikiTree_u.txt','r') as f:
    treestr = f.readlines()[0]    

t = Tree( treestr.rstrip(),format=8)

t.show()

## compare to

t = Tree( '("[a,b]",c);' )
t.show()

# []