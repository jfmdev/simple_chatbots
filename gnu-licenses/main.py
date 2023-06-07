# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:46:35 2023

@author: jofma
"""

# Import dependencies.
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer

from util import cosine_similarity, get_data

# Get data and train model.
data_df = get_data()

questions = data_df[0:10]['question'].to_numpy();
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sbert_model.encode(questions)

# Test model.
query = "All GNU software use the GNU GPL?"
query_vec = sbert_model.encode([query])[0]

best_value = 0
best_index = None
for index, question in enumerate(questions):
  sim = cosine_similarity(query_vec, sbert_model.encode([question])[0])
  print("\nQuestion = ", question, " |  Similarity = ", sim)
  
  if(best_value < sim):
    best_value = sim
    best_index = index

print(
  "\nBest answer:",
  data_df.iloc[[best_index]]['question'].item(),
  data_df.iloc[[best_index]]['answer'].item()
)