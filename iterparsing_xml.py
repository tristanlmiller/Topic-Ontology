# -*- coding: utf-8 -*-
"""
Created on 21 Jul

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

def xml_to_df(xmlname='enwiki-20160701-pages-articles-multistream.xml', nsample=100000):
    # converts MediaWiki xml to dataframe of indices, titles, and text 
    with open("iterparsing_xml.log", "a") as logfile:
        logfile.write(str(datetime.datetime.today())+'\n')
    indices = [int(index) for index in get_articles(xmlname)] # get indices of articles as list of ints
    random.seed(8685)
    indices = random.sample(indices,nsample)
    indices.sort()
    indices = [str(index) for index in indices] #converting to string, to be compared with xml
    titles, text, links = get_titles_text(xmlname,indices)
    df = get_dataframe(titles, text, links)
    df.to_pickle(xmlname+'.pkl')
    return df

def get_titles_text(xmlname,indices):
    titles = []
    text = []
    links = []
    for event, element in etree.iterparse(xmlname, tag="{http://www.mediawiki.org/xml/export-0.10/}page"):
        if element.find("{http://www.mediawiki.org/xml/export-0.10/}id").text in indices:
            title = element.find("{http://www.mediawiki.org/xml/export-0.10/}title")
            try:
                page = wiki.page(title.text)
                text.append(page.summary)
                titles.append(title.text)
                links.append(page.links)
            except Exception:
                with open("iterparsing_xml.log", "a") as logfile:
                    logfile.write(title.text + " : does not appear to have viable summary. Maybe it's a redirect?\n")
                pass
        element.clear()
    return titles, text, links

def get_articles(xmlname):
    indices = []
    for event, element in etree.iterparse(xmlname, tag="{http://www.mediawiki.org/xml/export-0.10/}page"):
        if element.find("{http://www.mediawiki.org/xml/export-0.10/}ns").text=='0':
            redirect = element.find('{http://www.mediawiki.org/xml/export-0.10/}redirect')
            if redirect is None:
                indices.append(element.find("{http://www.mediawiki.org/xml/export-0.10/}id").text) #yes, this will be a string corresponding to the id of the article
        element.clear()
    return indices

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
