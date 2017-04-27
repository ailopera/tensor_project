from __future__ import division, absolute_import, print_function

from six.moves import urllib, xrange

import tensorflow as tf
import numpy as np
import collections, math, os, random, zipfile

# Código sacado del tutorial: http://www-edlab.cs.umass.edu/~djsaunde/word2vec.html

# Download the data
url = 'http://mattmahoney.net/dc/'

######################
# Auxiliar Functions #
######################

def download(filename, expected_bytes):
	"""
	Download a file if not present, and make sure it's the right size.
	"""
	if not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
	if statinfo.st_size == expected_bytes:
		print('Found and verified', filename)
	else:
		print(statinfo.st_size)
		raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
	return filename

def read_data(filename):
	"""
	Parse the file enclosed in the 'filename' zip file into a list of words.
	"""
	# Unzip the file
	with zipfile.ZipFile(filename) as f:
		#Read the data into the 'data' variable
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()
	#Return the data
	return data

# FUNCTIONS TO BUILD THE WORD DICTIONARY
def build_dataset(words):
	# Create counts list, set counts for 'UNK' token to -1 (undefined)
	count = [['UNK', -1]]
	# add counts of the 49,999 most common tokens in 'words'
	# NOTE: The collections module implements specialized container datatypes providing alternatives to Python's general 
	# purpose built-in containers, dict, list, set, and tuple. Counter is a dict subclass for counting hashable objexts.
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	# Create the dictionary data structure
	dictionary = {}
	# Give a unique integer ID to each token in the dictionary
	# NOTE: Va recorriendo el count y le asigna un id a cada palabra, basado en el orden en el que se la ha encontrado
	#	Va construyendo un diccionario 	que contendrá los ids(indices) de cada palabra
	for word, _ in count:
		dictionary[word] = len(dictionary)
	# Create a list data structure for the data
	data = []
	
	# Keep track of the number of "UNK" token occurrences
	# NOTE: Contamos las palabras totales que no hemos tenido en cuenta 
	unk_count = 0
	# for each word in our dataset
	for word in words:
		# if its in the dictionary, get its index
		if word in dictionary:
			index = dictionary[word]
		#Otherwise, set the index equal to zero (index of "UNK") and increment the "UNK" count
		else:
			index = 0 # dictionary ['UNK']
			unk_count += 1
		# append its index to the 'data' list structure
		data.append(index)

	# Set the count of 'UNK' in the 'count' data structure
	count[0][1] = unk_count
	# Invert the dictionary; it becomes (idex, word) key-value pairs
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	# return the data (indices), counts, dictionary, and inverted dictionary
	return data, count, dictionary, reverse_dictionary


# Function that generates a training batch for the skip-gram model
# TODO: Mirar
def generate_bach(batch_size, num_skips, skip_window):
	global data_index
	# Make sure our parameters are self-consistent
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	# Create empty batch ndarray using 'batch_size'
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	# Create empty labels ndarray using 'batch_size'
	labels = np.array(shape=(batch_size, 1), dtype=np.int32)

	# [ skip_window target skip_window ]
	span = 2 * skip_window + 1
	# Create a buffer object for prepping batch data
	buffer = collections.deque(maxlen=span)
	
	# For each element in our calculated span, append the datum at 'data_index' 
	# and increment 'data_index' moduli the amount of data
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index +1) % len(data)
	
	# loop for 'batch_size' // 'num_skips'
	for i in range(batch_size // num_skips):
		#target label at the center of the buffer
		target = skip_window
		targets_to_avoid = [skip_window]
		# loop for 'num_skips'
		for j in range(num_skips):
			#loop through all 'targets_to_avoid'
			while target in targets_to_avoid:
				# pick a random index as target
				target = random.randint(0, span -1)
			
			# Put it in 'targets_to_avoid'
			targets_to_avoid.append(target)
			# set the skip window in the minibatch data
			batch[i * num_skips + j] = buffer[skip_window]
			# set the target in the minibatch labels
			labels[i * num_skips + j, 0] = buffer[target]

		
		# Add the data at the current 'data_index' to the bufer
		buffer.append(data[data_index])
		#increment 'data_index'
		data_index = (data_index + 1) % len(data)
	# Return the minibatch data and corresponding labels
	return batch, labels



##################
# Execution Code #
##################
# 1. Download the filename 
file_ = download('text8.zip', 31344016)
# 2. Read the data into the words variable
words = read_data(file_)
print('>>> Data size: ', len(words))

## Now that we've read in the raw text and converting it into a list of string, we'll need to convert this list into a dictionary
# of (input, output) pairs as described above for the Skip-Gram Model. We'll also replace rare words in the dictionary 
# with the token UNK, as is standard in this kind of NLP task.

# Output is the word from the context predicted. So a single input produces differents tuples of (input, output) 
# 3. Build the dictionary and replace rare words with the "UNK" token.
vocabulary_size = 50000
# Build the dataset
data, count, dictionary, reverse_dictionary = build_dataset(words)
# free up some memory
del words
#print out stats 
print('>>> Most common words (+UNK): ', count[:10])
print('>>> Sample data: ', data[:10], [reverse_dictionary[i] for i in data[:10]])


# 4. Generate minibatches for model training
# We use a function that allows us to generate mini-batches or training the skip-gram model.
data_index = 0
# Get a minibatch
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# Print out part of the minibatch to the console
for i in range(8):
	print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0], reverse_dictionary[labels[i, 0]])

# We set up some hyperparameters and set up the computation graph

# 5.  Hyperparameters
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector
skip_window = 1 # How many words to consider to left ad right
num_skips = 2 # How many times to reuse an input to generate a label

# We choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are alse the most frequently occuring words
valid_size = 16 #Size of random set of words to evaluate similarity on
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # Number of negative examples to sample


# Create computation graph
graph = tf.Graph()

with graph.as_default():
	# input data
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Operations and variables
	# Look up embeddings for inputs
	embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	#Construct the variables for the NCE loss
	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

	# Compute the average NCE loss for the batch.
	# tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
	loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases = nce_biases,
	labels=train_labels, inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))

	# Construct the SGD optimizer using a learning rate of 1.0
	optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

	# Compute the cosine similarity between minibatch examples and all embeddings
	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm
	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
	
	# Add variable initializer
	init = tf.initialize_all_variables()


