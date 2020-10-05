import pandas as pd
tweets = pd.read_csv('tweets.csv')
tweets.head()

import re

import string
def remove_noise(text):
    # Dealing with Punctuation
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

tweets['full_text'] = tweets['full_text'].apply(lambda x : remove_noise(x))

tweets['full_text'] = tweets['full_text'].apply(lambda x : remove_noise(x))

tweets.head()

tweets['full_text'] = tweets['full_text'].apply(lambda x : x.lower())

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

stop = stopwords.words('english')

def remove_stopwords(text):
    text = [item for item in text.split() if item not in stop]
    return ' '.join(text)

tweets['cleaned_data'] = tweets['full_text'].apply(remove_stopwords)

tweets.head()

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

tweets['stemed_text'] = tweets['cleaned_data'].apply(stemming)

tweets.head()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig, (ax1) = plt.subplots(1, figsize=[7, 7])
wordcloud = WordCloud( background_color='white', width=600, height=600).generate(" ".join(tweets['stemed_text']))

ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=16);

A=tweets['cleaned_data']

import pandas as pd
data = pd.DataFrame({'tweet_text':A})

data['tweet_text']

#Number of Words
data['word_count'] = data['tweet_text'].apply(lambda x: len(str(x).split(" ")))
data[['tweet_text','word_count']].head()

#Number of characters
data['char_count'] = data['tweet_text'].str.len() ## this also includes spaces
data[['tweet_text','char_count']].head()

#Average Word Length
#Number of characters(without space count)/Total number of words
def avg_word(sentence):
  words = sentence.split()
  print(words)
  print(len(words))
  print(sum(len(word) for word in words))
  return (sum(len(word) for word in words)/len(words))

data['avg_word'] = data['tweet_text'].apply(lambda x: avg_word(x))
data[['tweet_text','avg_word']].head()

#Number of stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['stopwords'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['tweet_text','stopwords']].head()

#Number of special characters
data['hastags'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['tweet_text','hastags']].head()

#Number of numerics
data['numerics'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['tweet_text','numerics']].head()

#Number of Uppercase words
data['upper'] = data['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['tweet_text','upper']].head()

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
from textblob import TextBlob, Word, Blobber
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
                print(ppo, tup)
    except:
        pass
    return cnt

data['noun_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'noun'))
data['verb_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'verb'))
data['adj_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'adj'))
data['adv_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'adv'))
data['pron_count'] = data['tweet_text'].apply(lambda x: check_pos_tag(x, 'pron'))
data[['tweet_text','noun_count','verb_count','adj_count', 'adv_count', 'pron_count' ]].head()

data.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv=CountVectorizer()
A_vec = cv.fit_transform(tweets['cleaned_data'])
print(A_vec.toarray())

tv=TfidfVectorizer()
t_vec = tv.fit_transform(tweets['cleaned_data'])
print(t_vec.toarray())

feature_names = tv.get_feature_names()

dense = t_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

feature_name

df_c =pd.concat([df,data], axis=1)
df_c

def ltoS(s):  
    str1 = ""    
    for e in s:  
        str1 += e   
    
    return str1  
        
B=ltoS(tweets["cleaned_data"])
print(B)
BigramsList = [] 

for i in range(len(B.split())-1):
  BigramsList.append((B.split()[i], B.split()[i+1]))
BigramsList
