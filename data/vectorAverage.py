# Ejecucion
# - Utilizando el modelo entrenado con nuestros datos: python vectorAverage.py 300features_10minwords_10contextALL
# - Utilizando modelo oficial: python vectorAverage.py ~/GoogleNews-vectors-negative300.bin

import sys, os, time
import csv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import gensim
from gensim.models import Word2Vec, KeyedVectors
from sklearn.preprocessing import Imputer
import word2VecModel
import textModelClassifier
from randomForestClassifier import randomClassifier

LOG_ENABLED = False

#  Calculamos la representación basada en el vector de medias de las palabras que aparecen en la review
# (si forman parte del vocabulario del modelo)
def makeFeatureVec(words, model, num_features, index2word_set, log=False):
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
        if counter%1000. == 0. and LOG_ENABLED:
            print("> Report", counter," of ", len(news))
        
        #log = True if counter == 100 else False
        #if log:
        #	print("> Report", counter," of ", len(news))
        # Call the function (defined above) that makes average feature vectors
        newsFeatureVecs[counter] = makeFeatureVec(report, model, num_features, index2word_set)
        counter = counter + 1

    # Version paralela del computo de vectores de caracteristicas
    # num_cores = multiprocessing.cpu_count()
    # newsFeatureVecs = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeFeatureVec)(report, model, num_features, index2word_set) for report in news)

    return newsFeatureVecs	


def makeWordList(text):
    wordList = word2VecModel.news_to_wordlist(text,remove_stopwords=True, clean_text=False)
    return wordList


def executeVectorAverage(word2vec_model, model_executed, binary, train_data=None, test_data=None, validation=False):
    basePath = "./fnc-1-original/aggregatedDatasets/"
    num_features = 300
    executionDesc = "vector_Average"
    # stances_model_name = sys.argv[1]
    # bodies_model_name = sys.argv[2]
        
    # model = KeyedVectors.load_word2vec_format(model_name)
    start = time.time()
    execution_start = start
    print("> Word2vec_model:", word2vec_model)
    print("> Binary: ", binary)
    
    if binary == True:
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    else:
        model = gensim.models.Word2Vec.load(word2vec_model)
        #model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model)
        #model = KeyedVectors.load_word2vec_format(word2vec_model)

    
    end = time.time()
    loadModelTime = end - start
    print("> Tiempo empleado en cargar el modelo: ", loadModelTime)
    

    # Cargamos los datos de entrenamiento y de test por defecto, si no se han especificado
    if train_data is None:
        trainDataPath = basePath + "train_data_aggregated.csv"
        trainData = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)
    else:
        trainData = train_data
    
    if test_data is None:
        testDataPath = basePath + "test_data_aggregated.csv"
        testData = pd.read_csv(testDataPath,header=0,delimiter=",", quoting=1)
    else:
        testData = test_data
    
    # Generamos representaciones de vectores de medias
    # En este caso si quitamos las stopwords, a diferencia a cuando creamos el modelo
    # Las stopwords pueden introducir ruido en el calculo de los vectores de medias
    
    print(">> Generating word2vec input model and applying vector average for train data...")
    clean_train_headlines = []
    clean_train_articleBodies = []
    trainDataVecs = {}
    num_cores = multiprocessing.cpu_count()

    start = time.time()
    clean_train_headlines = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in trainData['Headline'])	
    clean_train_articleBodies = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in trainData['ArticleBody'])
    end = time.time()
    
    trainDataFormattingTime = end - start
    print("> Time spent on formatting training data: ", trainDataFormattingTime)
    
    print(">> Getting feature vectors for train headlines...")
    start = time.time()
    trainDataVecsHeadline = getAvgFeatureVecs(clean_train_headlines, model, num_features)
    print(">> Getting feature vectors for train articleBodies...")
    trainDataVecsArticleBody = getAvgFeatureVecs(clean_train_articleBodies, model, num_features)
    end = time.time()
    trainFeatureVecsTime = end - start
    print(">> Time spent on getting feature vectors for training data: ", trainFeatureVecsTime)
    
    # Creamos un vector de 1x600 que contiene el titular y el cuerpo de noticia asociado, para alimentar el modelo de Machine Learning
    trainDataInputs = []
    for sample in zip(trainDataVecsHeadline, trainDataVecsArticleBody):
        trainSample = np.append(sample[0],sample[1])
        trainDataInputs.append(trainSample)
    
    # Hacemos lo mismo con los datos de test
    print(">> Generating word2vec input model and applying vector average for test data...")
    # testDataPath = basePath + "test_data_aggregated_mini.csv"

    clean_test_articleBodies = []
    clean_test_headlines = []
    testDataVecs = {}

    start = time.time()
    clean_test_headlines = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['Headline'])
    clean_test_articleBodies = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['ArticleBody'])
    end = time.time()
    testDataFormattingTime = end - start
    print(">> Time spent on formatting testing data: ", testDataFormattingTime)


    print(">> Getting feature vectors for test articleBodies...")
    start = time.time()
    testDataVecsArticleBody = getAvgFeatureVecs(clean_test_articleBodies, model, num_features)
    print(">> Getting feature vectors for test headlines...")
    testDataVecsHeadline = getAvgFeatureVecs(clean_test_headlines, model, num_features)
    end = time.time()
    testDataFeatureVecsTime = end - start
    print(">> Time spent on getting feature vectors for training data...", testDataFeatureVecsTime)
    
    # Creamos un vector de 1x600 que contiene el titular y el cuerpo de noticia asociado, para alimentar el modelo de Machine Learning
    testDataInputs = []
    for sample in zip(testDataVecsHeadline, testDataVecsArticleBody):
        testSample = np.append(sample[0],sample[1])
        testDataInputs.append(testSample)

    print("> Tamaño de los datos de entrada (entrenamiento): ", trainData.shape)
    print("> Tamaño de los datos de entrada (test): ", testData.shape)
    
    # Llamamos al clasificador con los datos compuestos
    start = time.time()
    classification_results = {}
    if model_executed == 'MLP':
        trainDataInputs = Imputer().fit_transform(trainDataInputs)
        testDataInputs = Imputer().fit_transform(testDataInputs)
        # Modelo basado en un MultiLayer Perceptron
        classification_results = textModelClassifier.modelClassifier(np.array(trainDataInputs), trainData['Stance'], np.array(testDataInputs), testData['Stance'])
    elif model_executed == 'RF':
        # Modelo basado en un randomForest sencillo
        trainDataInputs = Imputer().fit_transform(trainDataInputs)
        testDataInputs = Imputer().fit_transform(testDataInputs)
        classification_results = randomClassifier(np.array(trainDataInputs), trainData['Stance'], np.array(testDataInputs), testData['Stance'])
    else:
        print(">>> ERROR: No se ha ejecutado ningún modelo")
    end = time.time()
    modelExecutionTime = end - start
    execution_end = end
    totalExecutionTime = execution_end - execution_start
    print("> Time spent on fiting and predicting: ", modelExecutionTime)
    print(">> Metrics: ", classification_results)

    # Ponemos en un csv los tiempos de ejecucion para compararlos más adelante
    # Se genera un fichero por dia
    date = time.strftime("%Y-%m-%d")
    validationDesc = "validation" if validation else ""
    output_file = executionDesc + "_execution_" + date + validationDesc + ".csv"
    fieldNames = ["date", "executionDesc", "textModelFeatures", "modelName", "loadModelTime", \
        "trainDataFormattingTime","trainDataFeatureVecsTime","testDataFormattingTime","testDataFeatureVecsTime", "totalExecutionTime",\
        "trainInstances", "testInstances", "modelTrained", "modelExecutionTime","trainAccuracy", "testAccuracy",\
        "confusionMatrix", "averagePrecision", "recall"]
    
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
        executionData = {
         "date": time.strftime("%Y-%m-%d %H:%M"),
         "executionDesc": executionDesc, 
         "textModelFeatures": np.array(trainDataInputs).shape[1], 
         "modelName": model_name,
         "loadModelTime": loadModelTime,
         "trainDataFormattingTime": trainDataFormattingTime,
         "trainDataFeatureVecsTime": trainFeatureVecsTime,
         "testDataFormattingTime": testDataFormattingTime,
         "testDataFeatureVecsTime": testDataFeatureVecsTime,
         "totalExecutionTime": totalExecutionTime,
         "trainInstances": trainData.shape[0],
         "testInstances": testData.shape[0],
         "modelTrained": model_executed,
         "modelExecutionTime": modelExecutionTime,
         "trainAccuracy": classification_results["train_accuracy"],
         "testAccuracy": classification_results["test_accuracy"],
         "confusionMatrix": classification_results["confusion_matrix"],
         "averagePrecision": classification_results["average_precision"],
         "recall": classification_results["recall"]
         }
         
        newFile = os.stat(output_file).st_size == 0
        if newFile:
            writer.writeheader()
        writer.writerow(executionData)
        print(">> Stats exported to: ", output_file)


if __name__ == "__main__":
    model_name = sys.argv[1]
    model_executed = sys.argv[2] # Puede ser "MLP" "RF"
    binary = sys.argv[3]
    executeVectorAverage(model_name, model_executed, binary)
    
