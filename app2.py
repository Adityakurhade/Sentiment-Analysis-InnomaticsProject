# -*- coding: utf-8 -*-
"""
Created on Sun May 29 15:07:24 2022

@author: arkur
"""
from keras.models import load_model
#import numpy as np
import tensorflow as tf
import streamlit as st
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from string import punctuation
from bs4 import BeautifulSoup

import pickle
from keras_preprocessing.sequence import pad_sequences


st.title("Sentiment Analysis of Fine Food Review :memo: ")
st.subheader("with Pretrained Word2Vec")

st.write("""
         This Web app will find out sentiment of the review.
         We know that helpfulness increases with positivity of the review.
         """)
         
#st.subheader("Provide your review here and we will predict sentiment")

model = load_model("best_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl",'rb'))

# stop words
stop_words = pickle.load(open("stopwords.pkl",'rb'))


lemmatizer = WordNetLemmatizer()

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
    
    # full string
    text = " ".join(text)
    
    text.strip()
        
    return text

text = st.text_input("Enter your review here")

st.write("Your Review :" + text)

maxlen = 200
#sentiment = []
sentm =''
if len(text)>0:
    text = [preprocess_text(text)]
    #st.write(text)
    text_seq = tokenizer.texts_to_sequences(text)
    text_pad =  pad_sequences(text_seq, maxlen=maxlen)
    
    #st.write(text_pad)
    result = model.predict(text_pad)
    #st.write(result)
    #sentiment = 
    if result[0][0]>0.50:
        sentm = "Positive :innocent:"
    else:
        sentm = "Negative :persevere:"

pred = st.button("Predict Sentiment")
if pred == True:
    st.write("Sentiment of above review is")
    st.subheader(sentm)
    if sentm == "Positive :innocent:":
        st.write("And this review was also helpful for other customers.")
    elif sentm == "Negative :persevere:":
        st.write("And this review was not much helpful for the other customers.")
else:
    st.write("sentiment of your text will appear here")