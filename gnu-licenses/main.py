# -*- coding: utf-8 -*-
"""
Main application file.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

# Import dependencies and define constants.
import nltk
nltk.download('punkt')
from sentence_transformers import SentenceTransformer

from colorama import just_fix_windows_console as colorama_init
from colorama import Fore
colorama_init()

from util import cosine_similarity, get_data, get_embeddings

SOURCE_URL = 'https://www.gnu.org/licenses/gpl-faq.html'

# Load data and model.
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

data_df = get_data()
questions = data_df['question'].to_numpy()
embeddings = get_embeddings(sbert_model, questions)

# Print welcome message and wait for questions.
print(f'{Fore.CYAN}')
print("Hello! What do you want to know about GNU Licenses?")
print("(write 'exit' once you run out of questions)")
print(f'{Fore.RESET}')

query = ''
while query != 'exit':
  # Check that query isn't empty.
  if query.strip() != '':
    # Find the answer with the most similar question.
    query_vec = sbert_model.encode([query])[0]
    similarities_df = data_df.apply(
      # CAVEAT: the 'name' attribute indicates the 'index' of the row.
      lambda row: cosine_similarity(query_vec, embeddings[row.name]),
      axis=1
    )
    best_index = similarities_df.idxmax()

    # Print result.
    best_row = data_df.iloc[best_index] 
    best_answer = best_row['answer']
    best_question = best_row['question']
    best_source = SOURCE_URL + '#' + best_row['id']
    print(f'\n{Fore.YELLOW}Chatbot:{Fore.RESET} {best_answer}')
    print(f'(source: {Fore.MAGENTA}{best_source}{Fore.RESET} | {best_question})')
    
  # Ask for next question.
  query = input(f'\n{Fore.GREEN}You:{Fore.RESET} ')
