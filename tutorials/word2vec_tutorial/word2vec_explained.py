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


#Function that plots the resulting embedding trained
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
	# Set plot size in inches 
	plt.figure(figsize=(18,18))
	# Loop through all labels
	for i, label in enumerate(labels):
		# get the embedding vectors
		x, y = low_dim_embs[i,:]
		# plot them in a scatterplot
		plt.scatter(x, y)
		#annotations
		plt.annotate(label, xy=(x,y), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
	# Save the figure out
	plt.savefig(filename)


##################
# Execution Code #
##################
# 1. Download the filename 
print(">>> STEP 1. Downloading data...")
file_ = download('text8.zip', 31344016)
# 2. Read the data into the words variable
words = read_data(file_)
print(">>> STEP 2. Reading the data into the words variable...")
print('>>> Data size: ', len(words))

## Now that we've read in the raw text and converting it into a list of string, we'll need to convert this list into a dictionary
# of (input, output) pairs as described above for the Skip-Gram Model. We'll also replace rare words in the dictionary 
# with the token UNK, as is standard in this kind of NLP task.

# Output is the word from the context predicted. So a single input produces differents tuples of (input, output) 
# 3. Build the dictionary and replace rare words with the "UNK" token.
print(">>> STEP 3: Building the dictionary and replacing rare words...")
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
print(">>> STEP 4. Generate minibatches for model training.")
data_index = 0
# Get a minibatch
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

# Print out part of the minibatch to the console
for i in range(8):
	print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0], reverse_dictionary[labels[i, 0]])

# We set up some hyperparameters and set up the computation graph

# 5.  Hyperparameters (parámetros del algoritmo de aprendizaje)
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector
skip_window = 1 # How many words to consider to left ad right
num_skips = 2 # How many times to reuse an input to generate a label

# We choose random validation dataset to sample nearest neighbors
# here, we limit the validation samples to the words that have a low
# numeric ID, which are also the most frequently occuring words
valid_size = 16 #Size of random set of words to evaluate similarity on
valid_window = 100 # Only pick development samples from the first 'valid_window' words
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 # Number of negative examples to sample


# Create computation graph
# NOTE: A graph contains a set of tf-Operation objects, which represent units of computation;
# and tf.Tensor objects, which represetn the units of data that flow between operations
graph = tf.Graph()

#NOTE: Graph.as_default is a context manager, which overrides the current default graph for the lifetime of the context
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


# 6. Training the model
# Steps to train the model
print(">>> STEP 6. Training the model")
num_steps = 100001
with tf.Session(graph=graph) as session:
	# We must initialize all variables before using them
	init.run()
	print('initialized.')

	# Loop thiugh all training steps and keep track of loss
	average_loss = 0
	for step in xrange(num_steps):
		# generate a minibatch of training data
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
		
		# We perform a single update step by evaluating the optimizer operation (including it
		# in the list of returned values of session.run())
		_, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		# print average loss every 2,000 steps
		if step % 2000 == 0:
			if step > 0:
				average_loss /= 2000
			# The average loss is an stimate of the loss over the last 2000 batches.
			print("Average loss at step ", step, ": ", average_loss)
			print("-------------------------------------------------")
			average_loss = 0
		
		# Computing cosine similarity (expensive!)
		if step % 10000 == 0:
			sim = similarity.eval()
			for i in xrange(valid_size):
				# get a single validation sample
				valid_word = reverse_dictionary[valid_examples[i]]
				# Number of nearest neighbors 
				top_k = 8
				# Computing nearest neighbors
				nearest = (-sim[i,:]).argsort()[1:top_k + 1]
				log_str = "nearest to %s:" % valid_word
				for k in xrange(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = "%s %s," % (log_str, close_word)
				print(log_str)
				print("#######################################")
	final_embeddings = normalized_embeddings.eval()

# 7. Visualizing the embeddings
try:
	# import t-SNE and matplotlib.pyplot
	from sklearn.manifold import TSNE
	import matplotlib.pyplot as plt
	
	%matplotlib inline
	# Create the t-SNE object with 2 components, PCA initialization, and 5000 iterations
	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
	# plot only so many words in the embedding
	plot_only = 500
	# fit the TSNE dimensionality reduction technique to the word vector embedding
	low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
	# get the words associated with the points
	labels = [reverse_dictionary[i] for i in xrange(plot_only)]
	# call the plotting function
	plot_with_labels(low_dim_embs, labels)
except ImportError:
	print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
