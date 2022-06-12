# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:21:48 2022

@author: arkur
"""

import numpy as np
import streamlit as st
from pickle import load
# NLP related libraries
#import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
from bs4 import BeautifulSoup




st.title("Sentiment Analysis of Fine Food Review :memo: ")

st.write("""
         This Web app will find out sentiment of the review.
         
         We know that helpfulness increases with positivity of the review
         """)
         
st.subheader("Provide your review here and we will predict sentiment")
         

# stop words    
stop_words = set(stopwords.words("english"))
add_words = ['the', 'I', 'and', 'a', 'to', 'of', 'is', 'it',
             'for', 'in', 'this','that', 'my', 'with', 'but', 
             'have', 'was', 'are', 'you']

stop_words = stop_words.union(add_words)

# instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

w2v_model = load(open("w2v_model.pkl",'rb'))

classifier = load(open("xgboost_model_hyp.pkl",'rb'))

# creating vector for each review by averaging word vectors
def sent_vector(sent):
    sent = [word for word in sent if word in w2v_model.wv.index_to_key]
    sent_vec = np.mean(w2v_model.wv.__getitem__(sent), axis=0)
    return sent_vec


def preprocess_text(text):
      
    #removing links
    text = re.sub(r"http\S+","",text) 
    
    #removing html tags and keeping only texts
    text = BeautifulSoup(text,'lxml').get_text() 
    
    # removing words containing numeric digits
    text = re.sub(r"\S*\d\S*","", text).strip() 
    
    #removing non-alphabetic characters
    text = re.sub(r"[^a-zA-Z]+"," ", text) 
    
    # converting words with characters appearing more than 2 times to the normal meaningful words
    text = re.sub(r"(.)\1+",r"\1\1",text)
    
    # converting to lower case and creating list of tokenized words
    text = word_tokenize(text.lower())
    
    # removing stop words
    text = [word for word in text if not word in stop_words]
    
    # removing punctuations
    text = [word for word in text if word not in punctuation ]
    
    #lemmatization (obtaining verb form of word)
    text = [lemmatizer.lemmatize(word,'v') for word in text] 
        
    return text


text = st.text_input("Enter your review here")

st.write("Your Review :" + text)

sentiment= []
sentm=""
if len(text)>0:
    text = preprocess_text(text)
    
    text = sent_vector(text)
    
    arr = np.array(list(text))
    result = classifier.predict(arr.reshape(1,-1))
    
    if result[0]== 1:
        sentiment.append("Positive :innocent:")
    else:
        sentiment.append("Negative :persevere:")
        
    sentm = sentiment[0]
    

pred = st.button("Predict sentiment")

if pred == True:
    st.write("Sentiment of above review is")
    st.subheader(sentm)
    if sentm == "Positive :innocent:":
        st.write("And this review was also helpful for other customers")
    elif sentm == "Negative :persevere:":
        st.write("And this review was not much helpful for the other customers")
else:
    st.write("sentiment of your text will appear here")