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
dat=data['clean_tweets']

def tokenization_w(words):
    w_new = []
    for w in (words[:]):
        w_token = word_tokenize(w)
        if w_token != '':
            w_new.append(w_token)
    return w_new

dataset_sent=tokenization_w(dat)
word_len=len(dataset_sent)
dataset_sent

import itertools
dataset_words = list(itertools.chain(*dataset_sent))
dataset_words

# number of sentences
len(dataset_sent)

# total number of words
len(dataset_words)

# size of vocabulary
len(set(dataset_words))

#Divide the dataset into two Train data and Test data. Trigram Probabilities are learnt on the Train data and to evaluate it, we use the Test data.
#Out of 10000 sentences, 8000 sentences are used for training.

data_sents = dataset_sent[:8000]
data_sents_test = dataset_sent[8000:]

# number of words in train data
num_words = 0
for sentence in data_sents:
  num_words += len(sentence)
num_words

# create two lists containing words
data_words_train = dataset_words[:num_words]
data_words_test = dataset_words[num_words:]

#Method to generate a list of trigrams and bigrams and to get frequencies of unigrams, bigrams and trigrams



def createTrigram(data):
	listOfTrigrams = []
	listOfBigrams = []
	trigramCounts = {}
	bigramCounts = {}
	unigramCounts = {}

	for i in range(len(data)):
		if i < len(data) - 2:
			listOfTrigrams.append((data[i], data[i+1], data[i+2]))
			if (data[i],data[i+1],data[i+2]) in trigramCounts:
				trigramCounts[(data[i],data[i+1],data[i+2])] += 1
			else:
				trigramCounts[(data[i],data[i+1],data[i+2])] = 0
		
		if i < len(data) - 1:

			listOfBigrams.append((data[i], data[i + 1]))

			if (data[i], data[i+1]) in bigramCounts:
				bigramCounts[(data[i], data[i + 1])] += 1
			else:
				bigramCounts[(data[i], data[i + 1])] = 1

		if data[i] in unigramCounts:
			unigramCounts[data[i]] += 1
		else:
			unigramCounts[data[i]] = 1
		

	return listOfBigrams,listOfTrigrams, unigramCounts, bigramCounts, trigramCounts

#Method to learn trigram probabilities

def calcTrigramProb(listOfBigrams, listOfTrigrams, unigramCounts, bigramCounts, trigramCounts):

	listOfBigramProb = {}
	listOfTrigramProb = {}
	for bigram in listOfBigrams:
		word1 = bigram[0]
		word2 = bigram[1]
		
		listOfBigramProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))

	for trigram in listOfTrigrams:
		word1 = trigram[0]
		word2 = trigram[1]
		word3 = trigram[2]

		listOfTrigramProb[trigram] = trigramCounts.get(trigram) / bigramCounts.get((word1,word2))

	file = open('trigramProb.txt', 'w')
	file.write('Trigram' + '\t\t\t\t\t\t' + 'Count' + '\t' + 'Probability' + '\n')

	for trigrams in listOfTrigrams:
		file.write(str(trigrams) + ' : ' + str(trigramCounts[trigrams]) + ' : ' + str(listOfTrigramProb[trigrams]) + '\n')
	file.close()

	return listOfTrigramProb

#Method to learn trigram and bigram probabilities with Add-1 Smoothing

def trigramWithAddOneSmoothing(listOfBigrams, listOfTrigrams, unigramCounts, bigramCounts, trigramCounts):

	listOfBigramProb = {}
	listOfTrigramProb = {}
	cStarTrigram = {}
	cStarBigram = {}

	for bigram in listOfBigrams:
		word1 = bigram[0]
		word2 = bigram[1]
		listOfBigramProb[bigram] = (bigramCounts.get(bigram)+1)/(unigramCounts.get(word1)+len(unigramCounts))
		cStarBigram[bigram] = (bigramCounts[bigram]+1) * unigramCounts[word1] / (unigramCounts[word1] + len(unigramCounts))

	for trigram in listOfTrigrams:
		word1 = trigram[0]
		word2 = trigram[1]
		word3 = trigram[2]
		listOfTrigramProb[trigram] = (trigramCounts.get(trigram) + 1)/(bigramCounts.get((word1,word2)) + len(bigramCounts))
		cStarTrigram[trigram] = (trigramCounts[trigram] + 1) * bigramCounts[(word1,word2)] / (bigramCounts[(word1,word2)] + len(bigramCounts))

	file = open('addOneSmoothing.txt', 'w')
	file.write('Trigram' + '\t\t\t' + 'Count' + '\t' + 'Probability' + '\n')

	for trigrams in listOfTrigrams:
		file.write(str(trigrams) + ' : ' + str(trigramCounts[trigrams])
				   + ' : ' + str(listOfTrigramProb[trigrams]) + '\n')

	file.close()

	return listOfTrigramProb, cStarTrigram, listOfBigramProb, cStarBigram

# Main Program

# Create a list of trigrams and bigram and get frequencies of unigrams, bigrams and trigrams
listOfBigrams,listOfTrigrams, unigramCounts, bigramCounts, trigramCounts = createTrigram(data_words_train)

# Calculate trigram probabilities
trigramProb = calcTrigramProb(listOfBigrams, listOfTrigrams, unigramCounts, bigramCounts, trigramCounts)

# Apply Add-1 Smoothing and calculate probabilities and get reconstructed count of trigrams
trigramAddOne, addOneTrigramCstar, bigramAddOne, addOneBigramCstar = trigramWithAddOneSmoothing(listOfBigrams, listOfTrigrams, unigramCounts, bigramCounts, trigramCounts)

#Input a sentence and generate trigrams

input = 'we like natural langauge processing'

inputList = [] # list to store bigrams

for i in range(len(input.split())-2):
  inputList.append((input.split()[i], input.split()[i+1], input.split()[i+2]))
inputList

#Get Triigram counts and probabilities use them to calculate probability of the input sentence

# Open a file to write output
output1 = open('trigramProb-OUTPUT.txt', 'w')

# initial probability of a sentence
outputProb1 = 1

output1.write('Trigram\t\t\t\t\t\t\t' + 'Count\t\t\t\t' + 'Probability\n\n')

for i in range(len(inputList)):

  # if trigram is present in the model, get updated probability
  if inputList[i] in trigramProb:
    # write trigram, its count and probability to the file
    output1.write(str(inputList[i]) + '\t\t' + str(trigramCounts[inputList[i]]) + '\t\t' + str(trigramProb[inputList[i]]) + '\n')
    # multiply with probability of a current trigram
    outputProb1 *= trigramProb[inputList[i]]

  # if trigram is not present in the model, sentence probability is zero
  else:
    output1.write(str(inputList[i]) + '\t\t\t' + str(0) + '\t\t\t' + str(0) + '\n')
    outputProb1 *= 0

output1.write('\n' + 'Probablility = ' + str(outputProb1))
outputProb1

# Open a file to write output
output2 = open('addOneSmoothing-OUTPUT.txt', 'w')

# initial probability of a sentence
outputProb2 = 1

output2.write('Trigram\t\t\t\t\t\t\t' + 'Count\t\t\t\t' + 'Probability\n\n')

for i in range(len(inputList)):

  # if trigram is present in the model, get updated probability
  if inputList[i] in trigramAddOne:
    # Update probability of the sentence
    outputProb2 *= trigramAddOne[inputList[i]]

    output2.write(str(inputList[i]) + '\t\t' + str(addOneTrigramCstar[inputList[i]]) + '\t\t' + str(trigramAddOne[inputList[i]]) + '\n')

  # if trigram is not present in the model, use unigram counts to get estimated probability
  else:
    # if first word in a trigram is not present in unigrams, add with with count 1
    if inputList[i][0] not in unigramCounts:
      unigramCounts[inputList[i][0]] = 1
      # calculate probability of that word
      prob = (1) / (unigramCounts[inputList[i][0]] + len(unigramCounts))
      addOneTrigramCstar = 1 * unigramCounts[inputList[i][0]] / (unigramCounts[inputList[i][0]] + len(unigramCounts))

    # if a bigram is not present in bigrams, add with with count 1
    if (inputList[0][1],inputList[i][1]) not in bigramCounts:
      bigramCounts[(inputList[0][1],inputList[i][1])] = 1
      prob = (1) / (bigramCounts[(inputList[0][1],inputList[i][1])] + len(bigramCounts))
      #reconstructed count for the trigram
      addOneTrigramCstar = 1 * bigramCounts[(inputList[0][1],inputList[i][1])] / (bigramCounts[(inputList[0][1],inputList[i][1])] + len(bigramCounts))

# input sentence
print(input)

# list of bigrams
print(inputList)

# probability given by simple bigram model
print ('Trigram Model: ', outputProb1)

# probability given by bigram model with add-1 smoothing
print ('Add One: ', outputProb2)

def sentence_prob_with_next_word(next_word):
  outputProb = 1
  new_trigram = (input.split()[-2],input.split()[-1], next_word)
  if new_trigram in trigramAddOne:
    outputProb *= trigramAddOne[new_trigram]
  else:
    if (new_trigram[0], new_trigram[1]) not in bigramCounts:
      bigramCounts[new_trigram[0], new_trigram[1]] = 1
    prob = (1) / (bigramCounts[new_trigram[0], new_trigram[1]]+ len(bigramCounts))
    outputProb *= prob
  return outputProb

input = 'the engineers are'
possible_words = ['cheated', 'happy', 'smart', 'afraid']

inputList = []
outputProb = 1

for i in range(len(input.split())-2):
  inputList.append((input.split()[i], input.split()[i+1], input.split()[i+2]))


for i in range(len(inputList)):

  if inputList[i] in trigramAddOne:
    outputProb *= trigramAddOne[inputList[i]]
  else:
    if (inputList[i][0],inputList[i][1]) not in bigramCounts:
      bigramCounts[(inputList[i][0],inputList[i][1])] = 1
    prob = (1) / (bigramCounts[(inputList[i][0],inputList[i][1])] + len(bigramCounts))
    outputProb *= prob

max_prob = 0
index_of_next_word = -1
for i, word in enumerate(possible_words):
  final_prob = outputProb * sentence_prob_with_next_word(word)
  if final_prob > max_prob:
    max_prob = final_prob
    index_of_next_word = i

print('Next Word:', possible_words[index_of_next_word])
print('Output Sentece:', input, possible_words[index_of_next_word])

input1 = 'the market is very happy these days'
input2 = 'market is the happy these very days'


inputList1 = []
inputList2 = []


outputProb1 = 1
outputProb2 = 1


for i in range(len(input1.split())-2):
  inputList1.append((input1.split()[i], input1.split()[i+1], input1.split()[i+2]))

for i in range(len(input2.split())-2):
  inputList2.append((input2.split()[i], input2.split()[i+1], input2.split()[i+2]))


for i in range(len(inputList1)):
  if inputList1[i] in trigramAddOne:
    outputProb1 *= trigramAddOne[inputList1[i]]
  else:
    if (inputList1[i][0],inputList1[i][1]) not in bigramCounts:
      bigramCounts[(inputList1[i][0],inputList1[i][1])] = 1
    prob1 = (1) / (bigramCounts[(inputList1[i][0],inputList1[i][1])] + len(bigramCounts))
    outputProb1 *= prob1


for i in range(len(inputList2)):
  if inputList2[i] in trigramAddOne:
    outputProb1 *= trigramAddOne[inputList2[i]]
  else:
    if (inputList2[i][0],inputList2[i][1]) not in bigramCounts:
      bigramCounts[(inputList2[i][0],inputList2[i][1])] = 1
    prob2 = (1) / (bigramCounts[(inputList2[i][0],inputList2[i][1])] + len(bigramCounts))
    outputProb2 *= prob2

print (input1, ':', outputProb1)
print (input2, ':', outputProb2)

input1 = 'its a peace of information'
input2 = 'its a piece of information'


inputList1 = []
inputList2 = []


outputProb1 = 1
outputProb2 = 1


for i in range(len(input1.split())-2):
  inputList1.append((input1.split()[i], input1.split()[i+1],input1.split()[i+2]))

for i in range(len(input2.split())-2):
  inputList2.append((input2.split()[i], input2.split()[i+1],input1.split()[i+2]))


for i in range(len(inputList1)):
  if inputList1[i] in trigramAddOne:
    outputProb1 *= trigramAddOne[inputList1[i]]
  else:
    if (inputList1[i][0],inputList1[i][1]) not in bigramCounts:
      bigramCounts[(inputList1[i][0],inputList1[i][1])] = 1
    prob1 = (1) / (bigramCounts[(inputList1[i][0],inputList1[i][1])] + len(bigramCounts))
    outputProb1 *= prob1


for i in range(len(inputList2)):
  if inputList2[i] in trigramAddOne:
    outputProb2 *= trigramAddOne[inputList2[i]]
  else:
    if (inputList2[i][0],inputList2[i][1]) not in bigramCounts:
      bigramCounts[(inputList2[i][0],inputList2[i][1])] = 1
    prob2 = (1) / (bigramCounts[(inputList2[i][0],inputList2[i][1])] + len(bigramCounts))
    outputProb2 *= prob2

print (input1, ':', outputProb1)
print (input2, ':', outputProb2)

#Perplexity of add-1 smoothed trigram model:
def calculate_bigram_sentence_probability(input):

  inputList = []
  outputProb = 1

  for i in range(len(input)-1):
    inputList.append((input[i], input[i+1]))

  for i in range(len(inputList)):
    if inputList[i] in bigramAddOne:
      outputProb *= bigramAddOne[inputList[i]]
    else:
      if inputList[i][0] not in unigramCounts:
        unigramCounts[inputList[i][0]] = 1
      prob = (1) / (unigramCounts[inputList[i][0]] + len(unigramCounts))
      outputProb *= prob

  return outputProb

def calculate_number_of_bigrams(sentences):
        bigram_count = 0
        for sentence in sentences:
            # remove one for number of trigrams in sentence
            bigram_count += len(sentence) - 1
        return bigram_count

def calculate_trigram_sentence_probability(input):

  inputList = []
  outputProb = 1

  for i in range(len(input)-2):
    inputList.append((input[i], input[i+1],input[i+2]))

  for i in range(len(inputList)):
    if inputList[i] in trigramAddOne:
      outputProb *= trigramAddOne[inputList[i]]
    else:
      if (inputList[i][0],inputList[i][1]) not in bigramCounts:
        bigramCounts[(inputList[i][0],inputList[i][1])] = 1
      prob = (1) / (bigramCounts[(inputList[i][0],inputList[i][1])] + len(bigramCounts))
      outputProb *= prob

  return outputProb

def calculate_number_of_trigrams(sentences):
        trigram_count = 0
        for sentence in sentences:
            # remove two for number of trigrams in sentence
            trigram_count += len(sentence) - 2
        return trigram_count

def calculate_bigram_perplexity(model, sentences):
    number_of_bigrams = calculate_number_of_bigrams(sentences)
    bigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        p = calculate_bigram_sentence_probability(sentence)
        if p != 0.0:
          a = math.log(p)
        else:
          a = 0
        bigram_sentence_probability_log_sum -= a
    return math.pow(2, bigram_sentence_probability_log_sum / number_of_bigrams)

def calculate_trigram_perplexity(model, sentences):
    number_of_trigrams = calculate_number_of_trigrams(sentences)
    trigram_sentence_probability_log_sum = 0
    for sentence in sentences:
        p = calculate_trigram_sentence_probability(sentence)
        if p != 0.0:
          a = math.log(p)
        else:
          a = 0
        trigram_sentence_probability_log_sum -= a
    return math.pow(2, trigram_sentence_probability_log_sum / number_of_trigrams)

import math
print("PERPLEXITY over Training Data with bigram:", calculate_bigram_perplexity(bigramAddOne, data_sents))
print("PERPLEXITY over Test Data with bigram:", calculate_bigram_perplexity(bigramAddOne, data_sents_test))
print("PERPLEXITY over Training Data with trigram:", calculate_trigram_perplexity(trigramAddOne, data_sents))
print("PERPLEXITY over Test Data with trigram:", calculate_trigram_perplexity(trigramAddOne, data_sents_test))
