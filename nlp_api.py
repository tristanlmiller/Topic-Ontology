"""
Created on Mon Jul 11
Major revisions Jul 27 & Jul 28
@author: Diya

processing data in pandas data frame

"""
import pickle
import pandas
import numpy
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords #assuming nltk is already installed
from nltk import PorterStemmer, WordNetLemmatizer
import os
import string
#import time
#from sklearn.feature_extraction.text import CountVectorizer

def proc_text(pklname):
    df = pickle.load(open(pklname,'rb'))
    df['proctext'] = df['text'].apply(lambda x: para_to_stems(x) )
    df['proclinks'] = df['links'].apply(lambda x: ' '.join([item.lower().replace("-","_").replace(" ", "_").translate(str.maketrans({key: None for key in '().'})) for item in x]))
    pickle.dump(df, open(os.path.splitext(pklname)[0]+'_nlp.pkl','wb'))
    stem2lemdict,words2lemdict = stem2lem(df)
    pickle.dump((stem2lemdict,words2lemdict), open(os.path.splitext(pklname)[0]+'_stem2lem.pkl','wb'))
    return df

def para_to_words( raw_text ):
    rev_text = BeautifulSoup(raw_text,"lxml").get_text()
    rev_text = re.sub("(=[.\n]*)","",rev_text)
    letters_only = re.sub("[^a-zA-Z]", " ", rev_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return meaningful_words

def para_to_stems(raw_text):
    meaningful_words = para_to_words(raw_text)
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

def stem2lem (df):
    alltext = ' '.join(df['text'].apply(lambda x: ' '.join(para_to_words(x)))).replace("\n",' ')
    alltext = ' '.join(set(alltext.split()))
    stem2lemdict = dict()
    words2lemdict = dict()
    for word in alltext.split():
        stem = PorterStemmer().stem(word)
        if stem not in stem2lemdict:
            lem = WordNetLemmatizer().lemmatize(word)
            stem2lemdict[stem] = lem
            words2lemdict[word] = lem
    return stem2lemdict, words2lemdict
