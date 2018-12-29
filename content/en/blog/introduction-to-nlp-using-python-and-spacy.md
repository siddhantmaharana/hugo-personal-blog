---
title: "Introduction to NLP Using Python and Spacy"
date: 2018-10-07T13:37:45-05:00
draft: false
---

This notebook contains code examples to get you started with Natural Language Processing in Python. Uses various modules of **NLTK** and **Spacy**

The codebase and the data can be found in [here](https://github.com/siddhantmaharana/NLP101)

## Contents:

### Part A: Text Retrieval and Pre-processing
1. Extraction and Conversion
2. Removing special characters and stopwords
3. Stemming/Lemmatizing and Tokenizing
4. Sentence Segmentation

### Part B: Feature Generation/Document Representation
1. POS tagging and Dependency Parsing
2. Named Entity recognition
3. TF, TF-IDF

### Part C: Modelling and Other NLP tasks
1. Text classification
2. Text Similarity
3. Topic Modelling
___

<h1><center>Part A: Text Retrieval and Pre-processing</center></h1>

## 1. Text Extraction and Conversion

NLP stands for Natural Language Processing, which is defined as the application of computational techniques to the analysis and synthesis of natural language and speech. 

For this exercise we will take some sample documents i,e class action complaints for violations of the securities law. The data files are already in '.txt' format and thus they dont need conversion.

Otherwise, the data from other varied sources sources have to extracted and subsequently converted to machine readable format or '.txt' files. Several libraries which can help in these tasks include:
1. pdfminer.six
2. textract


```python
## Importing Libraries
import numpy as np
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')
%matplotlib inline
```

### Reading the contents of the folder

To read the contents of the folder, various libraries such as os, glob can be used.

In our example, we have used the os library and then we navigate to the 'target' folder to read the documents and process them further.


```python
## getting the current path
cur_dir = os.getcwd()
# navigating to the data folder
data_path = os.path.join(cur_dir, "data")

# Iterating the contents of the folder
for root, sub_dir,files in os.walk(data_path):
    ## only reading the first file
    for i,f in enumerate(files[0:1]):
        file_path = os.path.join(data_path,f)
        print ("processing file no :", i+1,'\n')
        text = open(file_path,'r').read()
        ## Printing a sample text file
        print (text)
        
        

```

## 2. Removing special characters and stopwords(using NLTK)

NLTK is one of the significant libraries used in natural language processing and is also widely popular among researchers and developers. We will use various tools by NLTK to process the text and mine the information needed.
More about [NLTK](http://www.nltk.org/book/ch01.html) 

If we notice the above text, there are a lot of unwanted characters such as **puncuations, newlines** which we need to deal with before heading into further analysis. 


```python
import nltk
from nltk.corpus import words as english_words, stopwords
import re

## replacing the newlines and extra spaces
corpus = text.replace('\n', ' ').replace('\r', '').replace('  ',' ').lower()

## removing everything except alphabets
corpus_sans_symbols = re.sub('[^a-zA-Z \n]', '', corpus)

## removing stopwords
stop_words = set(w.lower() for w in stopwords.words())

corpus_sans_symbols_stopwords = ' '.join(filter(lambda x: x.lower() not in stop_words, corpus_sans_symbols.split()))
print (corpus_sans_symbols_stopwords)
```

## 3. Stemming and Lemmatizing(using NLTK)

Now that we got rid of the unnecessary characters in the text, we can focus on the words and try to represent them in a more general and standardized format

**Stemming:**  Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word.


**Lemmatization:** Lemmatization, on the other hand, is an organized & step by step procedure of obtaining the root form of the word, it makes use of vocabulary (dictionary importance of words) and morphological analysis (word structure and grammar relations).


```python
from nltk.stem import PorterStemmer
stemmer=nltk.PorterStemmer()
corpus_stemmed = ' ' .join (map(lambda str: stemmer.stem(str), corpus_sans_symbols_stopwords.split()))
print (corpus_stemmed)
```

### Checking the word distribution in the document 

Now that we have a relatively cleaner corpus, lets try to vizualize the **top occuring terms** in the corpus.



```python
# Plot top 20 frequent words
from collections import Counter
word_freq = Counter(corpus_stemmed.split(" "))
import seaborn as sns
sns.set_style("whitegrid")
common_words = [word[0] for word in word_freq.most_common(20)]
common_counts = [word[1] for word in word_freq.most_common(20)]


plt.figure(figsize=(15, 12))

sns_bar = sns.barplot(x=common_words, y=common_counts)
sns_bar.set_xticklabels(common_words, rotation=45)
plt.title('Most Common Words in the document')
plt.show()
```


![png](/images/nlp1.png)


## Introducing Spacy

spaCy by explosion.ai is a library for advanced Natural Language Processing in Python and Cython. spaCy comes with pre-trained statistical models and word vectors, and currently supports tokenization for 20+ languages. It features the fastest syntactic parser in the world, convolutional neural network models for tagging, parsing and named entity recognition and easy deep learning integration. It's commercial open-source software, released under the MIT licence.

More can be found [here](https://spacy.io/usage/)

### Spacy features

1. **Tokenization:** Segmenting text into words, punctuations marks etc. 
2. **Dependency Parsing:** Assigning syntactic dependency labels, describing the relations between individual tokens, like subject or object.
3. **Lemmatization:** Assigning the base forms of words. For example, the lemma of "was" is "be", and the lemma of "rats" is "rat".
4. **Sentence Boundary Detection (SBD):**Finding and segmenting individual sentences.
5. **Named Entity Recognition (NER):** Labelling named "real-world" objects, like persons, companies or locations.
6. **Part-of-speech (POS) Tagging:** Assigning word types to tokens, like verb or noun.

We can download other language models by running a code like below in your shell or terminal

`
python -m spacy download en_core_web_sm
`
#### A simple example in SPACY


```python
import spacy
## Spacy example 
nlp = spacy.load('en')

doc = nlp("Lets see what Spacy is capable of doing.")
for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.pos_,
        token.tag_
    ))

```

    Lets	0	let	False	NOUN	NNS
    see	5	see	False	VERB	VB
    what	9	what	False	NOUN	WP
    Spacy	14	spacy	False	PROPN	NNP
    is	20	be	False	VERB	VBZ
    capable	23	capable	False	ADJ	JJ
    of	31	of	False	ADP	IN
    doing	34	do	False	VERB	VBG
    .	39	.	True	PUNCT	.


## Preprocessing in Spacy

The object “nlp” is used to create documents, access linguistic annotations and different nlp properties.

The document is now part of spacy.english model’s class and is associated with a number of features and properties.

We would take the text for the first document and pass it to the spacy's nlp object. Now doc contains various linguistic features which can be accesses quite easily.

## Spacy operation in just a single line!


```python
## passing our text into spacy
doc = nlp(text)

## filtering stopwords, punctuations, checking for alphabets and capturing the lemmatized text
spacy_tokens = [token.lemma_ for token in doc if token.is_stop != True \
                and token.is_punct != True and token.is_alpha ==True]

```

## Plotting top 20 words 

In just one line, we were able to convert the entire text file to a list of **Tokens**. These tokens are individual words freed from the junk and the stopwords which occur in the English lexicon.


```python

word_freq_spacy = Counter(spacy_tokens)

# Plot top 20 frequent words

sns.set_style("whitegrid")
common_words = [word[0] for word in word_freq_spacy.most_common(20)]
common_counts = [word[1] for word in word_freq_spacy.most_common(20)]


plt.figure(figsize=(15, 12))

sns_bar = sns.barplot(x=common_words, y=common_counts)
sns_bar.set_xticklabels(common_words, rotation=45)
plt.title('Most Common Words in the document')
plt.show()
```


![png](/images/nlp2.png)


## 4. Sentence Segmentation

Sentence segmentation means the task of splitting up the piece of text by sentence.

We could do this by splitting on the . symbol, but dots are used in many other cases as well so it is not very robust because of the presence of period in other parts of the sentences. 

Still, let's give it a try to see the sample sentences produced




```python
text_str = ''.join(text.replace('\n',' ').replace('\t',' '))
sentences_split = text_str.split(".")
sentences_split[67]
```




    ' This amount represents an extraordinarily large  27 4'



The above sentence doesn't seem to be complete and coherent. 

Spacy on the other hand simplifies this task and is quite robust when it comes to sentence segmentation 
### Spacy sentence segmentation



```python
doc = nlp(text_str)
sentence_list = [s for s in doc.sents]
sentence_list[67]
```




    Hsi entered voting agreements with Battery Ventures under which they have agreed to vote their  18 shares in favor of the adoption of the Merger Agreement.



The above sentence produced by spacy in indeed perfect. 

<h1><center>Part B: Feature Generation/Document Representation</center></h1>

## 1. POS tagging and Dependency Parsing

Part-of-speech tagging is the process of assigning grammatical properties (e.g. noun, verb, adverb, adjective etc.) to words. Words that share the same POS tag tend to follow a similar syntactic structure and are useful in rule-based processes.

spaCy features a fast and accurate syntactic **dependency parser**, and has a rich API for navigating the tree. The parser also powers the sentence boundary detection, and lets you iterate over base noun phrases, or "chunks"

As an example, we will take the above sentence and feed that into the **pos** and **dependency parser**.



```python
spacy.displacy.render(sentence_list[67], style='dep',jupyter=True,options = {'compact':60})
pos_list = [(token, token.pos_) for token in sentence_list[67]]
```
![svg](/images/nlp3.svg)

## 2. Named Entity Recognition

Entity recognition is the process of classifying named entities found in a text into pre-defined categories, such as persons, places, organizations, dates, etc. 

spaCy uses a statistical model to classify a broad range of entities, including **persons, organisations, dates**.

Even newer entities can be trained and used on a corpus of documents.


Below, we will run the **NER** detection on a subset of the corpus from our text and also check the captured **Names** and **Organisations**


```python
text_ent_example ="In November 2009, unbeknownst to the Board, defendants Chen and Guasman began \
discussing a potential sale of the Company with Battery Ventures. Also in November 2009, the  \
Company received inquiries from other private equity firms expressing an interest in a transaction to. \
acquire the Company. However, before the Board considered the overtures from the private equity  firms,\
on January 27, 2010, Battery Ventures and RAE entered anon-disclosure agreement.\
That same day, defendants Chen, Guasman, and Hsi gave a presentation to Battery Ventures about the  \
Company\'s business and outlook. From the beginning of their discussions with Battery Ventures, \
 defendants Chen, Gausman, and Hsi favored Battery Ventures over other suitors because of its desire \
 to retain their services and the potential for an . equity position in the go-forward company."

doc = nlp(text_ent_example)
spacy.displacy.render(doc, style='ent',jupyter=True)
```


### Cleaning the text column using Spacy

We will go ahead and clean the text column so that we can form word-embeddings from the text and then make our data ready for modeling.

**Optimizing in Spacy**

Spacy ingests the text and performs all the operations such that the objects have all the linguistic features possible and this might a bit time consuming. Since, we wont be needing the POS, NER, DEP features from the text, we would skip these. For this, we need to redefine the Spacy class.




```python

class SpacyMagic(object):
    _spacys = {}

    @classmethod
    def get(cls, lang):
        if lang not in cls._spacys:
            import spacy
            cls._spacys[lang] = spacy.load(lang, disable=['tagger', 'ner','pos'])
        return cls._spacys[lang]
    
def run_spacy(text):
    nlp = SpacyMagic.get('en')
    doc = nlp(text)
    return doc

def clean_text(inp):
    spacy_text = run_spacy(inp)
    out_str= ' '.join ([token.lemma_ for token in spacy_text if token.is_stop != True and token.is_punct != True\
                        and token.is_alpha ==True])
    return out_str

for i in df.index:
    df.loc[i,'cleaned_text'] = clean_text(df.loc[i,'text'])
```

## Term Frequency Array from the data


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

list_corpus = df["cleaned_text"].tolist()
list_labels = df["label"].tolist()

vectorizer = CountVectorizer(stop_words='english')
tf = vectorizer.fit_transform(list_corpus)

featurenames = vectorizer.get_feature_names()

```

This produces an array of 6077 rows as features where each of the words are represented as columns.


```python
print(featurenames[1:20])
for doc_tf_vector in tf.toarray():
    print(doc_tf_vector)
    

```

**tf.toarray()** contains the array with the **term-frequency** for all the words in the corpus

### TF-IDF representation of the Document


```python
transformer = TfidfVectorizer(stop_words='english')
tfidf = transformer.fit_transform(list_corpus)
featurenames = vectorizer.get_feature_names()
print(featurenames[1:20])
for doc_tf_vector in tfidf.toarray():
    print(doc_tf_vector)
```

**tfidf.toarray()** contains the array with the **tf-idf score** for all the words in the corpus

<h1><center>Part C: Modelling and Other NLP tasks</center></h1>
    
## 1. Text Classification

We have created embeddings for the "cleaned text" and now each of the words represent a feature and we can use the labels to train our data based on these embeddings.




```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

list_corpus = df["cleaned_text"].tolist()
list_labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, \
                                                                                random_state=40)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)


## Plotting the confusion matrix

import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)

    return plt



```


### Logistic regression

Let's start with a logistic regression model to predict whether the SMS is a spam or ham.



```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)
```

### Evaulating the results


```python
cm = confusion_matrix(y_test, y_predicted_counts)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Spam','Ham'], normalize=False, title='Confusion matrix')
plt.show()
print(cm)
```


![png](/images/nlp4.png)


    [[964   3]
     [ 14 134]]



```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
```

    accuracy = 0.985, precision = 0.985, recall = 0.985, f1 = 0.985


The accuracy seems to be pretty good and other models can also be tried out to improve upon the model further

## 2 Text Similarity

Textual Similarity is a process where two texts are compared to find the Similarity between them.

The two most popular packages used are:
1. Levenshtein
2. Fuzzy wuzzy

` pip install python-Levenshtein`

`pip install fuzzywuzzy`


```python

```
