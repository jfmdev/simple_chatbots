# -*- coding: utf-8 -*-
"""
File with utility functions.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import json
import os
import string
from unidecode import unidecode

import numpy as np
import pandas as pd

import tensorflow as tf	
from nltk import word_tokenize


CACHE_FOLDER = "__mycache__"
INTENTS_PATH = "./data/intents.json"
MODEL_PATH = f'./{CACHE_FOLDER}/model.h5'


class Metadata:
  def __init__(
    self, 
    patterns, 
    vocabulary, 
    labels
  ):
    self.patterns = patterns
    self.vocabulary = vocabulary
    self.labels = labels


# Remove special characters from a text and split it in a list of words.
def tokenize_text(text):
  text = unidecode(text.lower())
  return [i for i in word_tokenize(text) if i not in list(string.punctuation)]

# Generates an array to represent a phrase.
def get_words_vector(word_list, vocabulary):
  vector = [0 for _ in range(len(vocabulary))]
  for word in word_list:
    if word in vocabulary:
      vector[vocabulary.index(word)] = 1
  return vector


# Generate matrix of labels for train the model.
def get_labels_train_matrix(patterns, labels_list):
  matrix = patterns['label'].apply(
    lambda label: get_words_vector([label], labels_list)
  ).to_numpy()
  
  return np.vstack(matrix)


# Generate matrix of words vectors for train the model.
def get_words_train_matrix(patterns, vocabulary):
  matrix = patterns['words'].apply(
    lambda word_list: get_words_vector(word_list, vocabulary)
  ).to_numpy()
  
  return np.vstack(matrix)


# Generate (or load) a model trained with the intents metadata.
def get_trained_model(metadata):
  # Check if the model is already available on the cache.
  if os.path.exists(MODEL_PATH):
    try:
      cache_model = tf.keras.models.load_model(MODEL_PATH)
      return cache_model
    except:
      pass
  
  # Parse metadata.
  unique_labels_count = len(metadata.labels)
  train_words = get_words_train_matrix(metadata.patterns, metadata.vocabulary)
  train_labels = get_labels_train_matrix(metadata.patterns, metadata.labels)
  
  # Train model
  model = tf.keras.Sequential()	

  model.add(tf.keras.layers.InputLayer(input_shape=(len(train_words[0]))))	
  model.add(tf.keras.layers.Dense(64, activation="relu"))	
  model.add(tf.keras.layers.Dense(64, activation="relu"))
  model.add(tf.keras.layers.Dense(unique_labels_count, activation="softmax"))
  	
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])	  
  model.fit(train_words, train_labels, epochs=1000, batch_size=256)	  
  model.save(MODEL_PATH)
  
  return model


# Load intents file and parse it.
def load_intents():
  intents = []
  metadata = None
  
  # Open intents file.
  with open(INTENTS_PATH, encoding='utf-8') as intents_file:
    # Get intents.
    intents_data = json.load(intents_file)
    intents = intents_data['intents']

    # Generate metadata.
    patterns = []
    words_list = []
    labels_list = []

    # Iterate intents to generate list of labels and words.
    for intent in intents:
      # Update list of (unique) labels.
      if intent['id'] not in labels_list:
        labels_list.append(intent['id'])
      
      # Iterate patterns.
      for pattern_text in intent['patterns']:
        pattern_words = tokenize_text(pattern_text)

        # Update list of (unique) words.
        for word in pattern_words:
          if word not in words_list:
            words_list.append(word)
 
        # Add pattern to list.
        patterns.append(pd.Series({
          "label": intent['id'],
          "words": pattern_words
        }))

    metadata = Metadata(pd.DataFrame(patterns), words_list, labels_list)  
  
  return intents, metadata
