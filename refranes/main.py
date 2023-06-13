# -*- coding: utf-8 -*-
"""
Main application file.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from random import randrange
import numpy as np

from colorama import just_fix_windows_console as colorama_init
from colorama import Fore
colorama_init()

from util import get_phrase_array, get_trained_model, load_intents

# Load intents and generate (or load) model.
intents, metadata = load_intents()
model = get_trained_model(metadata)	

# Print welcome message and wait for questions.
print(f'{Fore.CYAN}')
print("¡Hola! ¿Que te gustaría saber?")
print("(escribe 'adios' o 'chau' cuando ya no tengas más preguntas)")
print(f'{Fore.RESET}')

query = ''
while query != 'adios' and query != 'chau':
  # Check that query isn't empty.
  if query.strip() != '':
    # Make prediction.
    query_array = get_phrase_array(query.split(' '), metadata.unique_words)
    prediction = model.predict(np.array([query_array]))
    pred_index = np.argmax(prediction)

    # Select a response (randomly).
    response_index = randrange(0, len(intents[pred_index]['responses']))
    response = intents[pred_index]['responses'][response_index]

    # Print result.
    print(f'\n{Fore.YELLOW}Chatbot:{Fore.RESET} {response}')
    
  # Ask for next question.
  query = input(f'\n{Fore.GREEN}Tú:{Fore.RESET} ')
