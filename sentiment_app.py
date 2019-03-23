import string
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import operator
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import time
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_breast_cancer
from id3 import Id3Estimator,export_text
from id3 import export_graphviz
from sklearn import metrics

#define stopwords-stemmer & lemmatizing
stop = stopwords.words('english')
stemmer = SnowballStemmer("english")
lemmatizer = nltk.stem.WordNetLemmatizer()

# define function for feature list
def get_feature(df, number):
    
    feature_list = []
    # create an instance for tree feature selection
    tree_clf = ExtraTreesClassifier()

    # first create arrays holding input and output data

    # Vectorizing Train set
    cv = CountVectorizer(analyzer='word')
    x_train = cv.fit_transform(df['review'])

    # Creating an object for Label Encoder and fitting on target strings
    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    # fit the model
    tree_clf.fit(x_train, y)
    
    # Preparing variables
    importances = tree_clf.feature_importances_
    feature_names = cv.get_feature_names()
    feature_imp_dict = dict(zip(feature_names, importances))
    sorted_features = sorted(feature_imp_dict.items(), key=operator.itemgetter(1), reverse=True)
    indices = np.argsort(importances)[::-1]

    # Create the feature list
    for f in range(number):
        feature_list.append(sorted_features[f][0])
    
    return(feature_list)
    
#Importing dataset
filepath = input('File Import: ')

#-----------Preprocess-----------------
#define objectrs & splitting dataset to train set, test set & unsupported reviews
dfcsv = pd.read_csv(filepath, sep=',' ,skip_blank_lines=True, error_bad_lines=False , engine='python')
df_trainset = dfcsv.query( 'label != "unsup" & type =="train"')
df_test = dfcsv.query('type == "test"').sample(n=1000, axis=None)
df_unsup = dfcsv.query('label == "unsup"')#.sample(n=1000, axis=None)

#choose 1250 negative & 1250 positive reviews for train set
dfn = df_trainset.query('label =="neg"').sample(n=1250, axis=None)
dfp = df_trainset.query('label =="pos"').sample(n=1250, axis=None)
df_25 = pd.concat([dfn, dfp], ignore_index=True)

#cleaning data set
df_25['review'] = df_25['review'].str.replace('[^\w\s]',' ').str.findall('\w{3,}').str.join(' ').str.replace('\d+', ' ').str.lower().apply(lambda x: [item for item in str(x).split() if item not in stop])
df_test['review'] = df_test['review'].str.replace('[^\w\s]',' ').str.findall('\w{3,}').str.join(' ').str.replace('\d+', ' ').str.lower().apply(lambda x: [item for item in str(x).split() if item not in stop])

#Snowball_Stemmer
df_test['review'] = df_test['review'].apply(lambda x: [stemmer.stem(y) for y in x])
df_25['review'] = df_25['review'].apply(lambda x: [stemmer.stem(y) for y in x])

#Lemmatizer
df_test['review'] = df_test['review'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
df_25['review'] = df_25['review'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

#concat train &test set to get infos about reviews
df_all = pd.concat([df_25, df_test], ignore_index=True)
df_all['review'] = df_all['review'].str.join(' ') # Tokenized words to document

################################## Preparing dataframe for model ##############################

# Creating df_algo dataframe which will be used for hypothesis testing
df_algo = pd.concat([df_25, df_test], keys=['train', 'test'])
df_algo = df_algo.reset_index(col_level=1).drop(['level_1'], axis=1)



df_algo['review'] = df_algo['review'].str.join(' ') #tokenize words to document

################################### Removing non feature words ###############################

# Creating the feature word_list

word_list = get_feature(df_algo[['review', 'label']], 3500)


################################## Splitting with feature selection data ###############################a

# Vectorising selected data
vect_algo = TfidfVectorizer(stop_words='english', analyzer='word')
vect_algo.fit(df_algo.review)
Xf_train = vect_algo.transform(df_algo[df_algo['level_0'].isin(['train'])].review)
Xf_test = vect_algo.transform(df_algo[df_algo['level_0'].isin(['test'])].review)

# Encoding target data
# Creating an object and fitting on target strings
le = LabelEncoder()
#split data to train & test
yf_train = le.fit_transform(df_algo[df_algo['level_0'].isin(['train'])].label)
yf_test = le.fit_transform(df_algo[df_algo['level_0'].isin(['test'])].label)

#Apply NaiveB to data
#_NaiveB
nb_start_time = time.time()
clf_nb = MultinomialNB()
clf_nb.fit(Xf_train, yf_train)
# predict the outcome for testing data
predictions = clf_nb.predict(Xf_test)
# check the accuracy of the model
nb_proc_time = (time.time() - nb_start_time)
nb_acc = accuracy_score(yf_test, predictions)

expected = yf_train
predicted = clf_nb.predict(Xf_train)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

print("NaiveB",nb_acc,nb_proc_time)