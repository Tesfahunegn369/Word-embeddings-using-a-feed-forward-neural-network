# Import Python libraries and helper functions (in utils2) 
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re                                                           # Load the Regex-module
from collections import Counter

from utils2 import sigmoid, get_batches, compute_pca, get_dict
from training_util import initialize_model, cost_func_test
from utils2 import softmax_test
from model import forward_prop_test, back_prop_test
from training import gradient_descent_test, gradient_descent
from visualization import visualization

# Download sentence tokenizer
nltk.data.path.append('.')
nltk.download('punkt_tab') # You MUST use punkt_tab (nltk>=3.8.2), not punkt

# Load, tokenize and process the data
with open('shakespeare.txt') as f:
    data = f.read()                                                # Read in the data
data = re.sub(r'[,!?;-]', '.', data)                   # Punctuations are replaced by .
data = nltk.word_tokenize(data)                                    # Tokenize string to words
data = [ch.lower() for ch in data if ch.isalpha() or ch == '.']    # Lower case and drop non-alphabetical tokens
print("Number of tokens:", len(data), '\n', data[:15])             # print data sample

# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary: ", len(fdist))
print("Most frequent tokens: ", fdist.most_common(20))  # print the 20 most frequent words and their freq.

# get_dict creates two dictionaries, converting words to indices and vice versa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
print("Size of vocabulary: ", V)

# example of word to index mapping
print("Index of the word 'king' :  ", word2Ind['king'])
print("Word which has index 2743:  ", Ind2word[2743])

softmax_test()
forward_prop_test()
cost_func_test(data)
back_prop_test(data)
gradient_descent_test(data)

print("\nCongratulations! You have completed all tasks.")

## Additional example: Train the model using shakespeare.txt, and visualize the word embeddings
print("\nAdditional example: Train the model using shakespeare.txt, and visualize the word embeddings")
N = 100  # Dimension of hidden vectors
C = 2  # Window size
num_iters = 300
learning_rate = 0.1

W1, W2, b1, b2 = initialize_model(N, V)
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, C, num_iters, alpha=learning_rate)

# Check that the word embeddings with similar meanings are encoded closely.
visualization(W1, W2, word2Ind)  # You can rotate the figure while clicking your left mouse button

