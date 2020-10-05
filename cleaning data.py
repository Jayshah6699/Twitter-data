import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer

from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from gensim.models import Word2Vec                                   #For Word2Vec

data_path = "tweets.csv"
data = pd.read_csv(data_path)

# load text
filename = 'tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
nltk.download('stopwords')

# load text
filename = 'tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
print(words[:100])

# load text
filename = 'tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
print(words[:100])

#stopwords = nltk.corpus.stopwords.words('english')
stop = set(stopwords.words('english')) 
print(stop)


# load data
filename = 'tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])
# load data
filename = 'tweets.csv'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
# remove all tokens that are not alphabetic

print("\nAfter removing all tokens that are not alphabetic\n")
words = [word for word in tokens if word.isalpha()]
print(words[:100])
