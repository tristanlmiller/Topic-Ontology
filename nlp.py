"""
Created on Mon Jul 11

@author: Diya

processing data in pandas data frame

"""

import parsing_xml
import pandas
import numpy
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords #assuming nltk is already installed
from nltk import PorterStemmer
#import time
#from sklearn.feature_extraction.text import CountVectorizer

def proc_text(df = parsing_xml.xml_to_df()):
    df['links'] = df['text'].apply(lambda x: get_links(x) )
    df['categories'] = df['text'].apply(lambda x: get_categories(x) )
    df['process'] = df['text'].apply(lambda x: para_to_words(remove_links(x)) )
    #old version:
    #df['text_noim'] = df['text'].apply(lambda x: re.sub("Category:","",re.sub("\[\[File:[\w+\s+\S+]+\|","",x)))
    #df['links'] = df['text_noim'].apply(lambda x: re.findall('\[\[(.*?)\]',x))
    #df['process'] = df['text_noim'].apply(lambda x: para_to_words(x) )
    return df

def para_to_words( raw_text ):
    rev_text = BeautifulSoup(raw_text,"lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", rev_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    stemmed_words = [PorterStemmer().stem(w) for w in meaningful_words]
    return(" ".join( stemmed_words ))

def get_links( raw_text ):
    #Looks for strings starting with [[ and ending with ]], but with no brackets in the middle
    #Often these links have one or more | in the middle, but I ignore all text after the first |
    #Additionally, if there is a colon :, that indicates a link to an image or category, or possibly something else
    #so I ignore strings that have colons in them.
    links = re.findall("\[\[([^\[\]\|:]*)\|?[^\[\]:]*\]\]",raw_text)
    #put links in lower case
    links = [link.lower() for link in links]
    return links

def get_categories( raw_text ):
    return re.findall("\[\[Category:([^\[\]\|]*)\|?[^\[\]]*\]\]",raw_text)

def remove_links( raw_text ):
    #Here I'm trying to preserve the text that is in the links
    #step 1: remove links of format [[linked article|link text]], replacing them with link text
    text_nolinks = re.sub("\[\[[^\[\]\|:]*\|([^\[\]:]*)\]\]","\\1",raw_text)
    #step 2: remove links of format [[linked article]], replacing them with linked article title
    text_nolinks = re.sub("\[\[([^\[\]\|:]*)\]\]","\\1",text_nolinks)
    #step 3: The only [[ ]] expressions remaining are those with colons in them.  These are presumably
    #links to images, categories, or the like.  They are removed entirely.
    text_nolinks = re.sub("\[\[[^\[\]]*\]\]","",text_nolinks)
    #step 4: External links are of format [external_link link text].  These are replaced with link text
    text_nolinks = re.sub("\[\s*[^\[\]]+\s*([^\[\]]*)\]","\\1",text_nolinks)
    #step 5: remove anything in {{ }}.  These are usually used for references
    #do it twice, because sometimes you have something like {{text {{text}} text}}
    text_nolinks = re.sub("\{\{[^\{\}]*\}\}","",text_nolinks)
    text_nolinks = re.sub("\{\{[^\{\}]*\}\}","",text_nolinks)
    #step 6: remove any html tags inside < >
    #Ah, apparently this step is redundant with BeautifulSoup.
    #text_nolinks = re.sub("<[^<>]*>","",text_nolinks)
    return text_nolinks
