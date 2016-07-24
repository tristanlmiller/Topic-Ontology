# -*- coding: utf-8 -*-
"""
Created on 24 Jul

@author: Diya

run_this will parse the Wikipedia xml file,
find all the non-redirect articles,
take a reproducible random subset of 10000 articles,
get the titles and text of these articles,
and return a pandas dataframe
"""

#import xml.etree.ElementTree as ET
from lxml import etree
import matplotlib.pyplot as plt
import numpy
import random
import pandas as pd
import wikipedia as wiki
import sys
import datetime
import pickle

def api_to_df(titlefile='en_wikipedia_titles.pkl', nsample=100000): 
    with open("iterparsing_titles.log", "a") as logfile:
        logfile.write(str(datetime.datetime.today())+'\n'+titlefile+'\n')
    titles =  get_titles(titlefile) 
    random.seed(8685)
    titles = random.sample(titles, nsample)
    titles, text, links = get_text(titles)
    df = get_dataframe(titles, text, links)
    df.to_pickle(titlefile+'_df.pkl')
    return df

def get_text(titles):
    titles = []
    text = []
    links = []
    for title in titles: 
        try:
            page = wiki.page(title)
            text.append(page.summary)
            titles.append(title)
            links.append(page.links)
        except Exception:
            with open("iterparsing_titles.log", "a") as logfile:
                logfile.write(title.text + " : does not appear to have viable summary. Maybe it's a stub?\n")
            pass
    return titles, text, links

def get_titles(titlefile):
    titles = pickle.load(open(titlefile,"rb"))
    return titles

def get_dataframe(titles, text, links):
    data=pd.DataFrame(index=range(len(titles)), columns=['title','text','links'])
    data['title']=titles
    data['text']=text
    data['links']=links
    return data

"""A few additional functions to characterize the articles"""
def get_length(text):
    #to execute:
    #(char_count, log_char, word_count, log_word) = get_length(text)
    char_count = [len(a) for a in text]
    log_char = numpy.log10(char_count)
    word_count = [len(a.split()) for a in text]
    log_word = numpy.log10(word_count)
    plt.hist(log_word)
    plt.ylabel("Number articles")
    plt.xlabel("Log number words")
    return char_count, log_char, word_count, log_word
