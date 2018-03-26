# Ejecucion
# - Utilizando el modelo entrenado con nuestros datos: python vectorAverage.py 300features_10minwords_10contextALL
# - Utilizando modelo oficial: python vectorAverage.py ~/GoogleNews-vectors-negative300.bin

import sys
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
import word2VecModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 
import pandas as pd

from joblib import Parallel, delayed
import multiprocessing

import textModelClassifier

#  Calculamos la representación basada en el vector de medias de las palabras que aparecen en la review
# (si forman parte del vocabulario del modelo)
def makeFeatureVec(words, model, num_features, index2word_set):
	#Function to average all of the word vectors in a given paragraph
	#Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	
	# #Indexwords is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
	# index2word_set = set(model.wv.index2word)

	#Loop over each word in the review and, if it is in the model's vocabulary, 
	# add its feature vector to the total
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1
			featureVec = np.add(featureVec, model[word])
	
	featureVec = np.divide(featureVec, nwords)
	return featureVec



def getAvgFeatureVecs(news, model, num_features):
	# Dado un conjunto de noticias (cada una es una lista de palabras), calcula 
	# El vector de medias para cada una y devuelve un vector de dos dimensiones

	counter = 0

	# Reservamos espacio para un array de dos dimensiones, por velocidad
	newsFeatureVecs = np.zeros((len(news), num_features), dtype="float32")

	#Indexwords is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
	index2word_set = set(model.wv.index2word)
	
	# Iteramos sobre las noticias
	for report in news:
		# Print a status message every 1000th new
		if counter%100. == 0.:
			print("> Report", counter," of ", len(news))
		
		# Call thhe function (defined above) that makes average feature vectors
		newsFeatureVecs[counter] = makeFeatureVec(report, model, num_features, index2word_set)
		counter = counter + 1

	# Version paralela del computo de vectores de caracteristicas
	# num_cores = multiprocessing.cpu_count()
	# newsFeatureVecs = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeFeatureVec)(report, model, num_features, index2word_set) for report in news)

	return newsFeatureVecs	


def makeWordList(text):
	wordList =  word2VecModel.news_to_wordlist(text,remove_stopwords=True, clean_text=False)
	return wordList

if __name__ == "__main__":
	basePath = "./fnc-1-original/aggregatedDatasets/"
	num_features = 300
	model_name = sys.argv[1]
	# stances_model_name = sys.argv[1]
	# bodies_model_name = sys.argv[2]
	
	# model = KeyedVectors.load_word2vec_format(model_name)
	model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
	# bodies_model = Word2Vec.load(bodies_model_name)
	# stances_model = Word2Vec.load(stances_model_name)
	
	#Primero las convertimos en lista de palabras
	# trainDataPath = basePath + "train_data_aggregated_mini.csv"
	trainDataPath = basePath + "train_data_aggregated.csv"
	trainData = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)
	clean_train_headlines = []
	clean_train_articleBodies = []
	trainDataVecs = {}
	# En este caso si quitamos las stopwords, a diferencia a cuando creamos el modelo
	# Las stopwords pueden introducir ruido en el calculo de los vectores de medias
	#for report in trainBodies['Headline']:
	print(">> Generating word2vec input model and applying vector average for train data...")
	# for index,line in trainData.iterrows():
	# 	headline = line['Headline']
	# 	articleBody = line['ArticleBody']
	# 	clean_train_headlines.append(word2VecModel.news_to_wordlist(headline,remove_stopwords=True))
	# 	clean_train_articleBodies.append(word2VecModel.news_to_wordlist(articleBody,remove_stopwords=True))

	num_cores = multiprocessing.cpu_count()
	clean_train_headlines = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in trainData['Headline'])
	clean_train_articleBodies = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in trainData['ArticleBody'])

	# Cargamos el modelo de word2vec en un set
	#Indexwords is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
	# index2word_set = set(model.wv.index2word)

	print(">> Getting feature vectors for train headlines...")
	trainDataVecsHeadline = getAvgFeatureVecs(clean_train_headlines, model, num_features)
	print(">> Getting feature vectors for train articleBodies...")
	trainDataVecsArticleBody = getAvgFeatureVecs(clean_train_articleBodies, model, num_features)

	trainDataInputs = []
	for sample in zip(trainDataVecsHeadline, trainDataVecsArticleBody):
		# trainSample = sample[0].extend(sample[1])
		trainSample = np.append(sample[0],sample[1])
		trainDataInputs.append(trainSample)

	#  Escribimos en un fichero los datos de entrenamiento
	# Hacemos lo mismo con los datos de test
	print(">> Generating word2vec input model and applying vector average for test data...")
	# testDataPath = basePath + "test_data_aggregated_mini.csv"
	testDataPath = basePath + "test_data_aggregated.csv"
	testData = pd.read_csv(testDataPath,header=0,delimiter=",", quoting=1)
	clean_test_articleBodies = []
	clean_test_headlines = []
	testDataVecs = {}
	#for report in testBodies['Headline']:
	# for index,line in testData.iterrows():
	# 	headline = line['Headline']
	# 	articleBody = line['ArticleBody']
	# 	clean_test_headlines.append(word2VecModel.news_to_wordlist(headline,remove_stopwords=True))
	# 	clean_test_articleBodies.append(word2VecModel.news_to_wordlist(articleBody,remove_stopwords=True))

	clean_test_headlines = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['Headline'])
	clean_test_articleBodies = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['ArticleBody'])
	
	print(">> Getting feature vectors for test articleBodies...")
	testDataVecsArticleBody = getAvgFeatureVecs(clean_test_articleBodies, model, num_features)
	print(">> Getting feature vectors for test headlines...")
	testDataVecsHeadline = getAvgFeatureVecs(clean_test_headlines, model, num_features)

	testDataInputs = []
	for sample in zip(testDataVecsHeadline, testDataVecsArticleBody):
		# testSample = sample[0].extend(sample[1])
		testSample = np.append(sample[0],sample[1])
		testDataInputs.append(testSample)

	print("> Tamaño de los datos de entrada (entrenamiento): ", trainData)
	print("> Tamaño de los datos de entrada (test): ", testData.shape)
	
	# Llamamos al clasificador con los datos compuestos
	textModelClassifier.modelClassifier(trainDataInputs,trainData['Stance'],testDataInputs,testData['Stance'])