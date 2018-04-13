print("Training the random forest...")

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import textModelClassifier
from randomForestClassifier import randomClassifier
import time
import numpy as np

# Pequeña funcion auxiliar para explorar las features de una representacion
def getFeaturesInfo(features):
    print(">>> features.shape: ", features.shape) 

    # We can also print the counts of each word in the vocabulary
    #Sum up the counts of each vocabulary word
    dist = np.sum(features,axis=0)

    # For each on, print the vocabulary word and the number of times it appears in the training set
    print(">>> Count of occurrences of each word")
    for tag,count in zip(vocab,dist):
        print(count,tag)


def createBOWModel(bow_train_data, printLogs=False):
    MAX_FEATURES = 150
    print(type(bow_train_data)," | ",bow_train_data[1], type(bow_train_data[1]) )
    print(">>> Creating the bag of words...\n")

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=MAX_FEATURES)

    # Entrenamos el modelo de BOW con los datos de entrenamiento
    vectorizer.fit(bow_train_data)

    #Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    if printLogs:
        print(">>> Feature names (Vocabulary)", vocab)

    return vectorizer


def generateBOWModel(model_executed, train_data=[],test_data=[]):
    basePath = "./fnc-1-original/aggregatedDatasets/"

    # Paso 0: Cargamos los datasets de entrada por defecto
    print(">> Loading data...")
    if train_data == []: 
        inputTrainFile = basePath + "train_data_aggregated.csv"
        train_data = pd.read_csv(inputStancesPath,header=0,delimiter=",", quoting=1)
    if test_data == []:
        inputTestFile = basePath + "test_data_aggregated.csv"
        test_data = pd.read_csv(inputStancesPath,header=0,delimiter=",", quoting=1)
    
    
    # Paso 1: Creamos el modelo de Bag of words de las noticias
    print(">> Creating BOW model...")
    # El modelo de fit lo hacemos a partir de las noticias (cuerpos de noticia) y los headlines de entrenamiento
    # (vectorizer, train_data_features) = createBOW.createBOWModel(cleanTrainBodies)
    vectorizer = createBOWModel(cleanTrainBodies)

    # Paso 2: Obtenemos las representaciones de entrenamiento y de test y lo convertimos a un array numpy
    train_headlines = vectorizer.transform(train_data["Headline"]).toarray()
    train_articleBodies = vectorizer.transform(train_data["ArticleBody"]).toarray()
    train_data_features = []
    for sample in zip(train_headlines, train_articleBodies):
        train_sample = np.append(sample[0], sample[1])
        train_data_features.append(train_sample)

    test_headlines = vectorizer.transform(test_data["Headline"]).toarray()
    test_articleBodies = vectorizer.transform(test_data["ArticleBody"]).toarray()
    test_data_features = [] 
    for sample in zip (test_headlines, test_articleBodies):
        test_sample = np.append(sample[0], sample[1])
        test_data_features.append(test_sample)

    # Paso 2: Ejecutamos los modelos
    if model_executed == 'MLP':
		# Modelo basado en un MultiLayer Perceptron
        classification_results = textModelClassifier.modelClassifier(train_data_features, train_data['Stance'], test_data_features, test_data['Stance'])
    elif model_executed == 'RF':
        # Inferimos las representaciones con valores incorrectos (NaN, Inf)
        # trainDataInputs = Imputer().fit_transform(trainDataInputs)
		# testDataInputs = Imputer().fit_transform(testDataInputs)
		
        # Modelo basado en un randomForest sencillo
		trainDataInputs = Imputer().fit_transform(trainDataInputs)
		testDataInputs = Imputer().fit_transform(testDataInputs)
		randomClassifier(train_data_features, train_data['Stance'], test_data_features, test_data['Stance'])

    print(">> Classification Results: ", classification_results)