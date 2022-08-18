#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries

from keras_preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
import keras.utils as ku 

from tensorflow import keras
import keras_nlp

# set seeds for reproducability
from numpy.random import seed
import tensorflow as tf

tf.random.set_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# Load the dataset

curr_dir = 'archive/'
all_headlines = []
for filename in os.listdir(curr_dir):
    if 'Articles' in filename:
        article_df = pd.read_csv(curr_dir + filename)
        all_headlines.extend(list(article_df.headline.values))
        break

all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)


# In[3]:


# Dataset preprocessing

def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in all_headlines]
corpus[:10]


# In[4]:


# N-gram Tokenization

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
inp_sequences[:10]


# In[5]:


# Add sequence padding

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


# In[6]:


# Create model using Transformer Decoder & GRU Neural Network

embed_dim = 50
num_heads = 4
rnn_units = 512
maxlen = max_sequence_len
vocab_size = total_words

def create_model():
    inputs = keras.layers.Input(shape=(maxlen-1,), dtype=tf.int32)
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim, 
                                                            num_heads=num_heads, 
                                                            dropout=0.5)(embedding_layer)
    gru_layer = GRU(rnn_units, return_sequences=False)(decoder)

    flat_layer = Flatten()(gru_layer)
    
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(flat_layer)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="adam", 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model()
model.summary()


# In[7]:


# Training the Model

model.fit(predictors, label, epochs=20, batch_size=256, verbose=1)


# In[8]:


# Generating the text

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted,axis=1)
        
        output_word = ""
        
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


# In[9]:


print (generate_text("war news", 5, model, max_sequence_len))
print (generate_text("great mistake", 4, model, max_sequence_len))
print (generate_text("donald trump", 4, model, max_sequence_len))
print (generate_text("germany and austria", 4, model, max_sequence_len))
print (generate_text("president news", 5, model, max_sequence_len))
print (generate_text("the truth life is", 4, model, max_sequence_len))
print (generate_text("science and technology", 5, model, max_sequence_len))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




