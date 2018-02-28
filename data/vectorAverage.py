import sys
import numpy as np
import gensim
from gensim.models import Word2Vec, KeyedVectors
import word2VecModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#  Calculamos la representación basada en el vector de medias de las palabras que aparecen en la review
# (si forman parte del vocabulario del modelo)
def makeFeatureVec(words, model, num_features):
	#Function to average all of the word vectors in a given paragraph
	#Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	
	#Indexwords is a list that contains the names of the words in the model's vocabulary. Convert it to a set, for speed
	index2word_set = set(model.wv.index2word)

	#Loop over each word in the review and, if it is in the model's vocabulary, 
	# add its feature vector to the total
	for word in words:
		if word in index2word_set:
			nwords = nwords + 1
			featureVec = np.add(featureVec, model[word])
	
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec, nwords)
	return featureVec


def getAvgFeatureVecs(news, model, num_features):
	# Dado un conjunto de noticias (cada una es una lista de palabras), calcula 
	# El vector de medias para cada una y devuelve un vector de dos dimensiones

	counter = 0

	# Reservamos espacio para un array de dos dimensiones, por velocidad
	newsFeatureVecs = np.zeros((len(news), num_features), dtype="float32")

	# Iteramos sobre las noticias
	for report in news:
		# Print a status message every 1000th new
		if counter%1000. == 0.:
			print("> Report", counter," of ", len(news))
		
		# Call thhe function (defined above) that makes average feature vectors
		newsFeatureVecs[counter] = makeFeatureVec(report, model, num_features)
		
		counter = counter + 1
	return newsFeatureVecs


if __name__ == "__main__":
	basePath = "./fnc-1-original/"
	num_features = 300
	model_name = sys.argv[1]
	
	#model = KeyedVectors.load_word2vec_format(model_name)
	model = Word2Vec.load(model_name)
	
	#Primero las convertimos en lista de palabras
	trainBodiesPath = basePath + "train_stances.csv"
	trainBodies = pd.read_csv(trainBodiesPath,header=0,delimiter=",", quoting=1)
	clean_train_news = []
	# En este caso si quitamos las stopwords, a diferencia a cuando creamos el modelo
	# Las stopwords pueden introducir ruido en el calculo de los vectores de medias
	#for report in trainBodies['Headline']:

	for index,line in trainBodies.iterrows():
		report = line['Headline']
		clean_train_news.append(word2VecModel.news_to_wordlist(report,remove_stopwords=True))

	trainDataVecs = getAvgFeatureVecs(clean_train_news, model, num_features)

	# Hacemos lo mismo con los datos de test
	print("> Creating average feature vecs for test reviews")
	testBodies = basePath + "test_stances.csv"
	clean_test_news = []
	#for report in testBodies['Headline']:
	for index,line in testBodies.iterrows():
		report = line['Headline']
		clean_test_news.append(word2VecModel.news_to_wordlist(report,remove_stopwords=True))

	testDataVecs = getAvgFeatureVecs(clean_test_news, model, num_features)

	# Creamos un modelo de random fores con los datos de entrenamiento, usando 100 árboles
	#TODO: falta tener en cuenta los cuerpos de la noticia
	forest = RandomForestClassifier(n_estimators=100)
	print("> Fitting a random fores to labeled training data...")
	forest = forest.fit(trainDataVecs, train["Stance"])

	# Test & extract results
	result = forest.predict(testDataVecs)

	# Write the test results
	output = pd.Dataframe(data={"id": test["id"], "stance": result})
	output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

