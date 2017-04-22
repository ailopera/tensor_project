# Versión ligeramente modificada y explicada del tutorial de word2vec
from __future__ import division, absolute_import, print_function

from six.moves import urllib, xrange

import tensorflow as tf
import numpy as np
import collections, math, os, random, zipfile

## Getting the Data
url = 'http://mattmahoney.net/dc'

def download(filename, expected_bytes):
	"""
	Download a file if not present, and make sure it's the right size.
	"""
	in not os.path.exists(filename):
		filename, _ = urllib.request.urlretrieve(url + filename, filename)
	statinfo = os.stat(filename)
		
