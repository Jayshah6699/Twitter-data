import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob, Word, Blobber
import pandas as pd
import re

data = pd.read_csv("tweets.csv")
Rev_tweet=data["text"]
Rev_tweet

def preprocess(text):
    clean_data = []
    for x in (text[:]): 
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers
        new_text = new_text.lower() # lower case, .upper() for upper
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
                           "]+", flags=re.UNICODE)
        new_text = emoji_pattern.sub(r'', new_text)         
        if new_text != '':
            clean_data.append(new_text)
    return clean_data

data['clean_text']=preprocess(Rev_tweet)
data['clean_text']

data['clean_tweets'] = data['clean_text'].str.replace("[^a-zA-Z#]", " ")
data['clean_tweets'] = data['clean_tweets'].fillna('').apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
data['clean_tweets'] = data['clean_tweets'].fillna('').apply(lambda x: x.lower())

data['clean_tweets']

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.decomposition import TruncatedSVD
# If nltk stop word is not downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# tokenization
tokenized_doc = data['clean_tweets'].fillna('').apply(lambda x: x.split())
# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# de-tokenization
detokenized_doc = []
for i in range(len(data)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
data['clean_tweets'] = detokenized_doc

data['clean_tweets']

# TF-IDF vector
vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True)
X = vectorizer.fit_transform(data['clean_tweets'])
X.toarray()

X.shape

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)
lsa = svd_model.fit_transform(X)

#Documents - Topic vector
pd.options.display.float_format = '{:,.16f}'.format
topic_encoded_df = pd.DataFrame(lsa, columns = ["topic_1", "topic_2"])
topic_encoded_df["documents"] = data['clean_tweets']
display(topic_encoded_df[["documents", "topic_1", "topic_2"]])

# Features or words used as features 
dictionary = vectorizer.get_feature_names()

dictionary

# Term-Topic matrix
encoding_matrix = pd.DataFrame(svd_model.components_, index = ["topic_1","topic_2"], columns = (dictionary)).T

encoding_matrix
