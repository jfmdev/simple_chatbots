# -*- coding: utf-8 -*-
"""
File with utility functions.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import pickle
import zipfile

import numpy as np
import pandas as pd
import pyarrow.feather as feather


CACHE_FOLDER = "__mycache__"
DATA_PATH = f'./{CACHE_FOLDER}/data.fthr'
EMBEDDINGS_PATH = f'./{CACHE_FOLDER}/embeddings.pkl'
CSV_FILE = "simpsons_script_lines.csv"
CSV_PATH = f'./{CACHE_FOLDER}/{CSV_FILE}'
ZIP_PATH = "./data/simpsons_script_lines.zip"

BATCH_SIZE = 100;


# Calculates a modifier (for the cocine similarity) that increases the relevance
# of script lines that belong to the same episode and dialog than the last match.
def context_modifier(previous_match, current_row):
  if previous_match is not None:
    if previous_match['conversation_id'] == current_row['conversation_id']:
      return 1.05
    if previous_match['episode_id'] == current_row['episode_id']:
      return 1.03
  return 1


# Calculates the cocine similarity between two word vectors.
def cosine_similarity(u, v):
  return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Get the (parsed) data from GNU Frequently Askes Questions HTML file.
def get_data():
  # Check if the index is already available on the cache.
  if os.path.exists(DATA_PATH):
    try:
      cache_df = feather.read_feather(DATA_PATH)
      return cache_df
    except:
      pass

  # Decompress ZIP file (if CSV file wasn't previously extracted).
  if not os.path.exists(CSV_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip:
      zip.extract(CSV_FILE, f'./{CACHE_FOLDER}')
  
  # Parse CSV file and update cache.
  data_df = parse_csv_file(CSV_PATH)
  feather.write_feather(data_df, DATA_PATH)

  return data_df


# Calculates (or loads from the cache) the embeddings of a list of phrases.
def get_embeddings(sbert_model, phrases):
  # Check if the embeddings are already available on the cache.
  if os.path.exists(EMBEDDINGS_PATH):
    try:
      with open(EMBEDDINGS_PATH, "rb") as fIn:
        cache_embeddings = pickle.load(fIn)  
        return cache_embeddings
    except:
      pass
    
  # Generate embeddings in batches (due the high number of phrases).
  embeddings = None
  phrases_count = len(phrases)
  batches_count = np.ceil(phrases_count / BATCH_SIZE).astype(int)
  
  for i in range(0, batches_count):
    print(f'[calculating embeddings] --- {i+1} out of {batches_count}')

    start = i * BATCH_SIZE
    end = (i+1) * BATCH_SIZE
    if end > phrases_count:
      end = phrases_count

    # Encode batch.    
    batch = sbert_model.encode(phrases[start:end])
    
    # CAVEAT: We can't initialize the array before because we don't know the shape.
    if embeddings is None:
      embeddings = np.empty([phrases_count, batch.shape[1]])
      
    # Append batch result to full list.
    embeddings[start:end] = batch 

  # Store embeddings.
  with open(EMBEDDINGS_PATH, "wb") as fOut:
    pickle.dump(embeddings, fOut, protocol=pickle.HIGHEST_PROTOCOL)
  
  return embeddings


# Checks if a line is empty.
def is_empty(line):
  return (
    line.empty or 
    not line['speaking_line'] or
    pd.isna(line['spoken_words'])
  )


# Checks if two lines are from the same dialog (note that on the dataset the lines 
# from diferent conversations are usually separated by non-speaking lines).
def is_same_conversation(line_a, line_b):
  if is_empty(line_a) or is_empty(line_b):
    return False

  # NOTE: The cases in which the conversations continue despite the location
  # changes aren't being considered with this logic.
  return (
    line_a["episode_id"] == line_b["episode_id"] and
    line_a["location_id"] == line_b["location_id"]
  )


# Parses the content of a HTML file.
def parse_csv_file(file_path):
  # Read CSV file (note we are assuming tha the rows are already sorted).
  df = pd.read_csv(file_path, encoding="utf8") 
  
  # Identify lines of the same conversation (iterating is the only way).
  conversation_acc = 1
  df['conversation_id'] = 0
  prev_row = pd.Series(dtype='float64')

  for index, row in df.iterrows():
    if not is_empty(row):
      if not is_same_conversation(prev_row, row):
        conversation_acc = conversation_acc + 1

      df['conversation_id'].iat[index] = conversation_acc
      
    prev_row = row
  
  # Drop empty lines and reset index.
  df = df[df['conversation_id'] != 0]
  df = df.reset_index()
   
  # Identify the end of each conversation.
  df['is_end'] = False
  ends = df.groupby(['conversation_id'])['id'].max()
  df.loc[df['id'].isin(ends), 'is_end'] = True

  return df
