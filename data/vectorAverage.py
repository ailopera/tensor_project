import numpy as np

def makeFeatureVec(words, model, num_features):
	#Function to average all of the word vectors in a given paragraph
	#Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	
	#Indexwords is a list that contains the names of the words in the model's vocabulary. Conver it to a set, for speed
	index2word_set = set(model.index2word)

