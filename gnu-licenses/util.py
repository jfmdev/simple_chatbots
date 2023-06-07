# -*- coding: utf-8 -*-
"""
File with utility functions.
"""

import os
import zipfile

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pyquery import PyQuery as pq


CACHE_FOLDER = "__mycache__"
DATA_PATH = f'./{CACHE_FOLDER}/data.fthr'
HTML_FILE = "Frequently Asked Questions about the GNU Licenses.html"
HTML_PATH = f'./{CACHE_FOLDER}/{HTML_FILE}'
ZIP_PATH = "./data/gnu-faq.zip"


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

  # Decompress ZIP file (if HTML file wasn't previously extracted).
  if not os.path.exists(HTML_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip:
      zip.extract(HTML_FILE, f'./{CACHE_FOLDER}')
  
  # Read HTML file content.
  file_obj = open(HTML_PATH, "r", encoding="utf8")
  file_content = file_obj.read()

  # Parse file and update cache.
  data_df = parse_html_file(file_content)
  feather.write_feather(data_df, DATA_PATH)

  return data_df


# Parses the content of a HTML file.
def parse_html_file(content):
  entries = []
  doc = pq(content)

  questions = doc('dl.article > dt')
  for question in questions:
    question_copy = pq(question).clone()
    pq(question_copy).find('span').remove()
    question_text = pq(question_copy).text()
    question_id = pq(question).attr('id')
    
    answer = pq(question).next_until('dt')
    answer_text = pq(answer).text()
    
    entries.append(pd.Series({
      "id": question_id,
      "question": question_text,
      "answer": answer_text
    }))
  
  return pd.DataFrame(entries)
