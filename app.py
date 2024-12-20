import pandas as pd
import numpy as np 
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import pad_sequences
import streamlit as st

word_index = imdb.get_word_index()
mapping_index_to_words = { v:k for (k,v) in word_index.items() }

from tensorflow.keras.models import load_model
model = load_model('model.h5')

with open("tokenizer.pkl" ,"rb") as file:
    tokenizer = pickle.load(file)

def predict(model, tokenizer, text, max_sequence_len):
  token_list = tokenizer.texts_to_sequences([text])[0]
  if len(token_list) >= max_sequence_len:
    token_list = token_list[-max_sequence_len:]
  input_sequence = pad_sequences([token_list],padding = "pre",maxlen = max_sequence_len - 1)
  pred = model.predict(input_sequence,verbose = 0)
  predicted_word = np.argmax(pred, axis = 1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word:
      return word
  return None

st.title("NEXT WORD PREDICTOR")
st.write("heyy! I am Joseph Joestar")
user_inp = st.text_area("Tell me anything..")
if st.button("predict"):
    next_word = predict(model,tokenizer,user_inp,8)
    st.write(f"Wait..!! Your next word is gonna be.. '{next_word}'")

else:
    st.write("")
