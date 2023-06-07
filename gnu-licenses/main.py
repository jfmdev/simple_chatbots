# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:46:35 2023

@author: jofma
"""

# Import dependencies.
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer

from util import cosine_similarity, get_data, get_embeddings

# Load data and model (if need).
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

data_df = get_data()
questions = data_df['question'].to_numpy()
embeddings = get_embeddings(sbert_model, questions)

# Test model.
query = "All GNU software use the GNU GPL?"
query_vec = sbert_model.encode([query])[0]

best_value = 0
best_index = None
for index, question in enumerate(questions):
  question_vec = embeddings[index]
  sim = cosine_similarity(query_vec, question_vec)
  print("\nQuestion = ", question, " |  Similarity = ", sim)
  
  if(best_value < sim):
    best_value = sim
    best_index = index

print(
  "\nBest answer:",
  data_df.iloc[[best_index]]['question'].item(),
  data_df.iloc[[best_index]]['answer'].item()
)