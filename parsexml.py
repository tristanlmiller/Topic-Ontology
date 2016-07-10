# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 12:23:40 2016

@author: diyadas
"""
 
## from https://docs.python.org/3/library/xml.etree.elementtree.html#tutorial

import xml.etree.ElementTree as ET

tree = ET.parse('simplewiki-20160701-pages-articles-multistream.xml')
root = tree.getroot()

N = len(root.getchildren())-1
print('There are '+ N +' pages.')

pages = [child[0].text for child in root]


colons = [title for title in pages if ':' in title]
len(colons)

#for child in root:
    #print(child.tag, child.attrib)
    #print(child.attrib)
    
#for item in root.iter('neighbor'):
#    print(item)    
#    print(item.attrib)
    
