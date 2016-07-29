# Topic-Ontology

Project by Diya Das ([@InfH](https://github.com/InfH)), Tristan Miller ([@tristanlmiller](https://github.com/tristanlmiller/)), Ali Sanaei ([@asanaei](https://github.com/asanaei)), and Simon Segert ([@SimonSegert](https://github.com/simonsegert))

Our goal is to create a topic ontology of Wikipedia articles and place any English Wikipedia article within this classification scheme, based on its text content.

This project is an experiment using Python to parse text and run classification algorithms. Dependencies include [pandas](https://pypi.python.org/pypi/pandas), [NumPy](https://pypi.python.org/pypi/numpy), [lxml](https://pypi.python.org/pypi/lxml), [wikipedia](https://pypi.python.org/pypi/wikipedia/), [BeautifulSoup4](https://pypi.python.org/pypi/beautifulsoup4/4.5.0), and [NLTK](https://pypi.python.org/pypi/nltk/3.2.1). Note: we are using Python 3.5. This code has not been tested on Python 2 or older versions of Python 3.

#### Overview
1. We extract a list of all articles in English Wikipedia from one of the XML Wikipedia dumps located [here](https://dumps.wikimedia.org/enwiki/). We are using the dump from 07-01-2016, as more recent dumps had not been finished at the start of our project. Any MediaWiki-formatted XML should do. (We used a Simple Wikipedia dump as input for testing.)
2. We randomly sample to get a subset of these articles, excluding redirects and other non-"pages", for constructing our main tree. The choice of number of articles depends largely on processing constraints.
3. We extract summary text for the selected articles and perform natural language processing. We calculate tf-idf to weight the features and perform PCA to maximize information content. We select the first 400 PCs for use in the remaining analysis.  
4. We perform iterative clustering on the PCA-rotated data, to identify groups of textually similar articles and construct our hierarchy.
5. When presented with an article not in the tree, we will perform tf-idf and transpose the article vector into the PCA rotated space. We calculate the distance from the medoids of the clusters, and use this information to place the article in the most similar cluster.


#### Specific procedures
1. iterparsing_titles.py:
 - "api_to_df()" reads "en_wikipedia_titles.pkl.bz2" which is a pickled containing all titles in the English Wikipedia and produces a data frame containing title, text and links for these articles.

2. nlp_api.py:
 - "proc_text(pickle)" reads the above df and adds two more columns: a cleaned text (stemmed,...) , and another which cleans links and replaces white space in links with underscore.

3. bag_of_words.py:
 - "BagOfWords.pkl", a vectorized form of the text plus links, using TF-IDF weighting.
 - "Tfidf_Matrix.pkl", the TF-IDF weighting used.
 - "Feature_List.pkl", a list of the terms used in the vectorization.
 - "PCA_components.pkl", a dimensionally reduced form of BagOfWords, with #dim = 400.
 - "PCA_matrix.pkl", the matrix used for dimensional reduction.
 - Additionally, we apply the same analysis to just the links, producing "Link_BagOfWords.pkl", "Link_Tfidf_Matrix.pkl", "Link_Feature_List.pkl", "Link_PCA_components.pkl", and "Link_PCA_matrix.pkl"

4. Clustering: Two methods are currently being attempted

 a. Ward_clustering.py (to come soon): Ward Clustering: (k= # clusters)
  - c_labels: Nx1 array (N= # docs)
  - label_tree: binary tree with clusters as leaves
  - collapsed_tree -> tol~0 seems fine.
  - cluster_means -> kx400 array.
  - docs_in_cluster -> kx1 array.
  - link_means

 b. Iterative_Kmeans.py

5. Tree Visualization
 - Labeling Nodes: Most common link/word compared to adjacent branches.

6. t-SNE
7. Node Visualization
8. Classification of new articles.