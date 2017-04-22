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

# Functions to building the word dictionary



##################
# Execution Code #
##################
# 1. Download the filename 
file_ = download('text8.zip', 31344016)
# 2. Read the data into the words variable
words = read_data(file_)
print('data size: ', len(words))

## Now that we've read in the raw text and converting it into a list of string, we'll need to convert this list into a dictionary
# of (input, output) pairs as described above for the Skip-Gram Model. We'll also replace rare words in the dictionary 
# with the token UNK, as is standard in this kind of NLP task.


