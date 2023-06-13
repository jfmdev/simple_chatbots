# -*- coding: utf-8 -*-
"""
File with utility functions.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import json
import tensorflow as tf	
import numpy as np

CACHE_FOLDER = "__mycache__"
INTENTS_PATH = "./data/intents.json"
MODEL_PATH = f'./{CACHE_FOLDER}/model.h5'


class Metadata:
  def __init__(
    self, 
    all_words, 
    all_labels, 
    unique_words, 
    unique_labels
  ):
    self.all_words = all_words
    self.all_labels = all_labels
    self.unique_words = unique_words
    self.unique_labels = unique_labels


# Generates an array to represent a phrase.
def get_phrase_array(word_list, vocabulary):
  array = [0 for _ in range(len(vocabulary))]
  for word in word_list:
    if word in vocabulary:
      array[vocabulary.index(word)] = 1
  return array

  
# Generate matrix of labels for train the model.
def get_train_labels(all_labels, unique_labels):
  train_labels = []
  
  for label in all_labels:
    row = [0 for _ in range(len(unique_labels))]
    row[unique_labels.index(label)] = 1
    train_labels.append(row)
  
  return np.array(train_labels)


# Generate matrix of words for train the model.
def get_train_words(all_words, unique_words):
  train_words = []

  for word_list in all_words:
    row = get_phrase_array(word_list, unique_words)
    train_words.append(row)
  
  return np.array(train_words)


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
  unique_labels_count = len(metadata.unique_labels)
  train_words = get_train_words(metadata.all_words, metadata.unique_words)
  train_labels = get_train_labels(metadata.all_labels, metadata.unique_labels)
  
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
    all_labels = []
    all_words = []
    unique_words = []
    unique_labels = []

    # Iterate intents to generate list of labels and words.
    for intent in intents:
      if intent['id'] not in unique_labels:
        unique_labels.append(intent['id'])
      
      for pattern in intent['patterns']:
        all_labels.append(intent['id'])

        pattern_words = pattern.split(' ')
        all_words.append(pattern_words)

        for word in pattern_words:
          if word not in unique_words:
            unique_words.append(word)
  
    metadata = Metadata(all_words, all_labels, unique_words, unique_labels)  
  
  return intents, metadata
