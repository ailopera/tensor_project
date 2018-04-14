print("Training the random forest...")
import os, time, sys
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import textModelClassifier
from randomForestClassifier import randomClassifier

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
    executionDesc = "Bag Of Words"

    # Paso 0: Cargamos los datasets de entrada por defecto
    print(">> Loading data...")
    execution_start = time.time()
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
    start = time.time()
    train_headlines = vectorizer.transform(train_data["Headline"]).toarray()
    train_articleBodies = vectorizer.transform(train_data["ArticleBody"]).toarray()
    end = time.time()
    trainDataFeatureVecsTime = end - start

    train_data_features = []
    for sample in zip(train_headlines, train_articleBodies):
        train_sample = np.append(sample[0], sample[1])
        train_data_features.append(train_sample)

    start = time.time()
    test_headlines = vectorizer.transform(test_data["Headline"]).toarray()
    test_articleBodies = vectorizer.transform(test_data["ArticleBody"]).toarray()
    end = time.time()
    testDataFeatureVecsTime = end - start

    test_data_features = [] 
    for sample in zip (test_headlines, test_articleBodies):
        test_sample = np.append(sample[0], sample[1])
        test_data_features.append(test_sample)

    # Paso 2: Ejecutamos los modelos
    start = time.time()
    classification_results = {}
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
    end = time.time()
    modelExecutionTime = end - start
    execution_end = time.time()
    totalExecutionTime = execution_end - execution_start
    print(">> Classification Results: ", classification_results)

    # Ponemos en un csv los tiempos de ejecucion para compararlos más adelante
    # Se genera un fichero por dia
    date = time.strftime("%Y-%m-%d")
    output_file = "vectorAverage_execution_" + date + ".csv"
    fieldNames = ["date", "executionDesc", "textModelFeatures", "modelName", "loadModelTime", \
        "trainDataFormattingTime","trainDataFeatureVecsTime","testDataFormattingTime","testDataFeatureVecsTime", "totalExecutionTime",\
        "trainInstances", "testInstances", "modelTrained", "modelExecutionTime", "trainAccuracy", "testAccuracy",\
        "confusionMatrix", "averagePrecision", "recall"]
    
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
        executionData = {
         "date": time.strftime("%Y-%m-%d %H:%M"),
         "executionDesc": executionDesc, 
         "textModelFeatures": train_data_features.shape[1], 
         "modelName": "",
         "loadModelTime": "",
         "trainDataFormattingTime": 0,
         "trainDataFeatureVecsTime": trainDataFeatureVecsTime,
         "testDataFormattingTime": 0,
         "testDataFeatureVecsTime": testDataFeatureVecsTime,
         "totalExecutionTime": totalExecutionTime,
         "trainInstances": train_data.shape[0],
         "testInstances": test_data.shape[0],
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
    model_executed = sys.argv[1] # Puede ser "MLP" "RF"
    generateBOWModel(model_executed)
    