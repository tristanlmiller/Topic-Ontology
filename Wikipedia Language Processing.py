### Wikipedia Language Processing
"""
When we acquire the text from Wikipedia data, we need to transform it into
a bag of words, with more weight given to "topic-relevant" words.
"""

### Current version notes:
"""
Here I will write some procedures using practice text data, which I took from
the Bag of Words to Bag of Popcorn Kaggle competition.

Specifically, I am using "TestData.tsv" from
https://www.kaggle.com/c/word2vec-nlp-tutorial/data
only I've renamed it to "MovieReviews.tsv"

Later, we will have to adjust the code for Wikipedia.
 - Tristan Miller, 7/8/2016
"""

### Importing Packages
import pandas
import numpy
import matplotlib as plt
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords #assuming nltk is already installed
from nltk import PorterStemmer
import time
from sklearn.feature_extraction.text import CountVectorizer

### Initialization of movie review data
def initialize_movie_reviews():
    """Executes all initializating procedures below.
    Returns bag of words and vocabulary as tuple"""
    movie_reviews = get_movie_data()
    clean_reviews = clean_all_reviews(movie_reviews)
    return create_bag_of_reviews( clean_reviews )
    
def get_movie_data():
    "Loads text data from MovieReviews.  Output is a series"
    return pandas.read_csv("MovieReviews.tsv", header=0, delimiter="\t",
                           quoting=3)["review"]
                           
def review_to_words( raw_review ):
    "Takes a single movie review and converts to a string with stemmed words"
    #Remove html
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #Remove non-letters 
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    #Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #Extract stemmed words
    stemmed_words = [PorterStemmer().stem(w) for w in meaningful_words]
    return( " ".join( stemmed_words ))
    
def clean_all_reviews( raw_review_data ):
    "Takes a list of reviews and turns each review into a list of stemmed words"
    clean_reviews = []
    start = time.time()
    for i in range(len(raw_review_data)):
        clean_reviews.append(review_to_words(raw_review_data[i]))
    
    processing_time = (time.time() - start)/60
    print("Time to clean reviews: %.2f minutes" % processing_time)
    #On my laptop this takes 1.18 minutes -Tristan
    return clean_reviews

def create_bag_of_reviews( clean_reviews ):
    "Creates a bag of words from movie reviews.  Returns array and vocabulary as tuple"
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
    start = time.time()
    bag_of_reviews = vectorizer.fit_transform(clean_reviews)
    processing_time = (time.time() - start)/60
    print("Time to create bag of words: %.2f minutes" % processing_time)
    #Takes 0.05 minutes -Tristan

    bag_of_reviews = bag_of_reviews.toarray()
    return ( bag_of_reviews , vectorizer.get_feature_names() )
    