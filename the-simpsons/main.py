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

# Load data and model.
data_df = get_data()
script_lines = data_df['spoken_words'].to_numpy()

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = get_embeddings(sbert_model, script_lines)

# Print welcome message and wait for questions.
print(f'{Fore.CYAN}')
print("Hello! Ask anything and someone from Springfield will answer you")
print("(write 'exit' once you run out of questions)")
print(f'{Fore.RESET}')

query = ''
while query != 'exit':
  # Check that query isn't empty.
  if query.strip() != '':
    # Find the script line most similar to the user's question.
    query_vec = sbert_model.encode([query])[0]
    similarities_df = data_df.apply(
      # Ignore lines that ends the conversation.
      lambda row: 
        # CAVEAT: the 'name' attribute indicates the 'index' of the row.
        cosine_similarity(query_vec, embeddings[row.name]) 
        if not row['is_end']
        else 0,
      axis=1
    )
    match_index = similarities_df.idxmax()

    # The answer will be the next line to the matching line.
    match_row = data_df.iloc[match_index] 
    match_words = match_row['spoken_words']
    next_row = data_df.iloc[match_index + 1] 
    next_character = next_row['raw_character_text']
    next_words = next_row['spoken_words']
    episode = match_row['episode_id']
    
    print(f'\n{Fore.MAGENTA}{next_character}:{Fore.RESET} {next_words}')
    print(f'{Fore.CYAN}(Source = Episode: {episode} | Previous line: {match_words}){Fore.RESET}')
    
  # Ask for next question.
  query = input(f'\n{Fore.GREEN}You:{Fore.RESET} ')
