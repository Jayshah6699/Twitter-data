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
corpus=data['clean_tweets']
data['clean_tweets']

# we need to pass splitted sentences to the model
tokenized_sentences = [sentence.split() for sentence in corpus]
lines=tokenized_sentences

#importing the glove library
from glove import Corpus, Glove

# creating a corpus object
corpus = Corpus() 

#training the corpus to generate the co occurence matrix which is used in GloVe
corpus.fit(lines, window=10)

glove = Glove(no_components=40, learning_rate=0.05)
 
glove.fit(corpus.matrix, epochs=40, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove2582020.model')

print(glove.word_vectors[glove.dictionary['charlottesville']])

import warnings
warnings.filterwarnings('ignore')

from gensim.models.word2vec import Word2Vec
import gensim.downloader as api

!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove*.zip

!ls
#!pwd

import numpy as np

print('Indexing word vectors.')

embeddings_index = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embeddings_index['sumit']

ls

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('/content/glove.6B.100d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

model.most_similar('obama')

model.most_similar('banana')

model.most_similar(negative='banana')

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))
