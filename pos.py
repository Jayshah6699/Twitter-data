import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
text = "learned nlp from sandip sir"

tokens = nltk.word_tokenize(text)
print(tokens)
tag = nltk.pos_tag(tokens)
print(tag)
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp  =nltk.RegexpParser(grammar)
result = cp.parse(tag)
print(result)
