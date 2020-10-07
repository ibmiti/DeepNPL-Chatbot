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

# getting the length of each line from clean_answers list
# then adding the token to the end of each line by referncing the number found by getting the length of each line
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' 
    
# Translating all questions and answers into integers 
# And replacing all the words that were filtered out by <OUT>
questions_to_int = []    
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else: 
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)        

answers_to_int = []    
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else: 
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
    
# Sorting questions and answers by the length of questions 
sorted_clean_questions= []
sorted_clean_answers = []
for length in range(1, 25 + 1):
  for i in enumerate(questions_to_int):
      if len(i[1]) == length:
          sorted_clean_questions.append(questions_to_int[i[0]])
          sorted_clean_answers.append(answers_to_int[i[0]])
          
          
#### Building the SEQ2SEQ MODEL ####
# using this function to create tensorflow place holders
def model_inputs():
    # inputs represented by 2 dimensional list 
    inputs  = tf.placeholder(tf.int32,[None, None], name = 'input')              
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr      = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets 
def preprocess_targets(targets, word2int, batch_size):
    left_side  = tf.fill([batch_size, 1], word2int['<SOS>'])
    # [batch_size, - 1] means everything within batch_size except the last one
    right_side = tf.strided_slice(targets,[0,0], [batch_size, -1],[1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
    
# Creating the Encoder RNN Layer 
def encoder_rnn_layer(rnn_inputs, rnn_size, keep_prob, sequence_length):
#    long short term memory as lstm
    lstm             = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    ltsm_dropout     = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell     = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidrectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_length = sequence_length,
                                                      inputs = rnn_inputs,
                                                      dtype = tf.float32)
    return encoder_state

# Decoding the training set 
    # we are now decoding the encoded state...
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope    
                                                                                                              )
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


    
    
    
    
    
    
    
                    
    
    