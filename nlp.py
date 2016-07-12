"""
Created on Mon Jul 11

@author: Diya

processing data in pandas data frame

"""

import parsing_xml
import pandas
import numpy
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords #assuming nltk is already installed
from nltk import PorterStemmer
import time
from sklearn.feature_extraction.text import CountVectorizer

def proc_text(df = parsing_xml.xml_to_df()):
    df['text_noim'] = df['text'].apply(lambda x: re.sub("Category:","",re.sub("\[\[File:[\w+\s+\S+]+\|","",x)))
    df['links'] = df['text_noim'].apply(lambda x: re.findall('\[\[(.*?)\]',x))
    df['process'] = df['text_noim'].apply(lambda x: para_to_words(x) )
    return df

def para_to_words( raw_text ):
    rev_text = BeautifulSoup(raw_text,"lxml").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", rev_text) 
    words = letters_only.lower().split()         
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    stemmed_words = [PorterStemmer().stem(w) for w in meaningful_words]
    return(" ".join( stemmed_words ))
