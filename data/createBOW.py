from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cleanBodies

#INPUT_DIR = "./fnc-1-original/cleanDatasets/"
#INPUT_DATA_PATH = 'train_bodies_clean.csv'
cleanTrainBodies = cleanBodies.cleanTextBodies(False)
MAX_FEATURES = 5000
print("Creating the bag of words...\n")


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=MAX_FEATURES)


# fittransform() does two functions:
# - First, it fits the model an learns the vocabulary
# - Second, it trahnsforms our training data into feature vectors.
# The input to fit_transform should be a list of strings
train_data_features = vectorizer.fit_transform(cleanTrainBodies)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

print(">>> train_data_features.shape: ", train_data_features.shape) 

#Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print(">>> Feature names (Vocabulary)")

# We can also print the counts of each word in the vocabulary
#Sum up the counts of each vocabulary word
dist = np.sum(train_data_features,axis=0)

# For each on, print the vocabulary word and the number of times it appears in the training set
for tag,count in zip(vocab,dist):
    print count,tag