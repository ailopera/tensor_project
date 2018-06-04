import tensorfflow as tf

# We iitialize our embedding matrix to a big rando matrix.
#We'll initialize the values to be uniform in the unit cube

embeddings = tf.Variable(
	tf.random_uniform([ vocabulary_size, embedding_size], -1.0, 1.0))

# The noise-contrastive estimation loss is defined in terms of a logistic regression model.
# For this, we need to define the weights and biases for each word in the vocabulary (also 
# called the output weights as opposed to the input embeddings)
nce_weights = tf.Variable(
	tf.truncated_normal([vocabulary_size, embedding_size],
			stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Now that we have the parameters in place, we can define our skip-gram model graph.
# Let's suppose we've already integerized our text corpus with a vocabulary so that 
# each word is represented as an integer

# The skip-gram model takes two inputs. One is a batch full of integers representing 
# the source context words, the other is for the target words.

# Let's create placeholder nodes for these inputs, so that we can feed in data later.
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# Now what we need to do is look up the vector for each of the source words in the batch.
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Now that we have the embeddings for each word, we'd like to try to predict the target
# word using the noise-contrastive training objective

# Compute the NCE loss, using a sample of the negative labels each time
loss = tf.reduce_mean(
	tf.nn.nce_loss(weights=nce_weigths,
			biases=nce_biases,
			labels=train_labels,
			inputs=embed,
			num_sampled=num_sampled,
			num_clases=vocabulary_size))


# Now that we have a loss mode, we need to add te nodes required to compute gradients
# and update the parameters, etc, For this we will use stochastic gradient descent, and
# TensorFlow has handy helpers to make this easy as well

# We use the SGD optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

### Training the model ###
