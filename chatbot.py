# Building a chatbot with Deep NLP 

# importing libraries 
import numpy as np
import tensorflow as tf 
import re as r
import time as t

# -----------------------------Gaining access to data, as GAD  ---------------

lines = open('movie_lines.txt', encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# ------------------------------End of GAD, as EOGAD -----------------------------------

# ------------------------- Cleaning GAD, as Data -----------------------

# creating dictionary of line ids
id2line = {}
for line in lines :
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# creating list of all conversation
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    
# getting separately the questions and the answers    
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
            

