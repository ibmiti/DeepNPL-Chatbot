# Building a chatbot with Deep NLP 

# importing libraries 
import numpy as np
import tensorflow as tf 
import re as replace
import time as t

# -----------------------------Gaining access to data, as GAD  --------------

lines = open('movie_lines.txt', encoding='utf-8',errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# ------------------------------End of GAD, as EOGAD ------------------------

# ------------------------- Cleaning GAD, as Data ---------------------------

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
            
# Doing a first cleaning of the texts        
# the argument for this param will be either questions or answers         
def clean_text(text):
    text = text.lower()
    text = replace.sub(r"i'm", "i am", text)
    text = replace.sub(r"he's", "he is", text)
    text = replace.sub(r"she's", "she is", text)
    text = replace.sub(r"that's", "that is", text)
    text = replace.sub(r"what's", "what is", text)
    text = replace.sub(r"where's", "where is", text)
    text = replace.sub(r"\'ll", " will", text)
    text = replace.sub(r"\'ve", " have", text)
    text = replace.sub(r"\'re", " are", text)
    text = replace.sub(r"\'d", " would", text)
    text = replace.sub(r"won't", "will not", text)    
    text = replace.sub(r"can't", "cannot", text)
    text = replace.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text
    
    
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
    
word2count = {}
# for evert item within clean_question arr
for question in clean_questions:
    # split every word within and place each new string into a temp var of word
    for word in question.split():
        # if word not in word2count ( avoids duplicates )
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]= 1
        else:
            word2count[word] += 1
            
# creating two dictionaries that map the questions words and the answers to a unique integer
threshold = 20
questionswords2int = {}
word_number = 0

# for there are words that we want to take account of.
# and for there are a number of words we will be accounting for 
# these words and the count of them will be found within word2count dictionary 

# simplified :
# loop through the word2count dict. items and give each thing within into 2 variables 
# one will hold the value of the words the other will take acount of the integer value for each    
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0 
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1                

# Adding the last tokens to these two dictionaries 
tokens = ['<PAD>', '<EOS>', '<OUT>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1 

# inversing map of a dictionary 
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}        
    
# Adding the End Of String token to the end of every answer 
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' 
