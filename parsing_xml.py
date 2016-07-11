# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:17:53 2016

@author: Tristan

run_this will load the Wikipedia xml file as an ElementTree,
find all the non-redirect articles,
take a reproducible subset of 10,000 articles,
get the titles and text of these articles
"""

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy
import random

def run_this():
    global root
    global indices
    global titles
    global text
    """This section contains the code to be executed"""
    root = parse_file()
    indices = get_article_indices(root)
    random.seed(8685)
    indices = random.sample(indices,10000)
    indices.sort()
    
    titles = get_titles(root,indices)
    text = get_text(root,indices)
    """end section"""

def parse_file():
    tree = ET.parse('simplewiki-20160701-pages-articles-multistream.xml')
    return tree.getroot()

def get_titles(root,indices):
    return [root[i][0].text for i in indices ]
    
def get_article_indices(root):
    indices = []
    for i in range(1,len(root.getchildren())):
        # Remove non-articles (ie help pages, categories, etc.)
        if root[i][1].text == "0":
            # Remove redirect articles
            redirect = root[i].find('{http://www.mediawiki.org/xml/export-0.10/}redirect')
            if redirect is None:
                indices.append(i)
    return indices

def get_text(root,article_indices):
    text = []
    for child in [root[i] for i in article_indices]:
        for textnode in child.iter(tag ='{http://www.mediawiki.org/xml/export-0.10/}text'):
            text.append(textnode.text)
    return text