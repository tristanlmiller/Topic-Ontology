# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:24:47 2016

@author: diyadas

This script plots trees of our wikipedia data.

"""

from ete3 import Tree

with open('/Users/diyadas/cdips/Topic-Ontology/SimpleWikiTree.txt','rb') as f:
    treestr = f.read()    

t = Tree( treestr )

t.show()