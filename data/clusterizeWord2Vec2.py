from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import sys
import math
import gensim
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import textModelClassifier
from randomForestClassifier import randomClassifier

import multiprocessing
from joblib import Parallel, delayed

from sklearn.preprocessing import Imputer

from itertools import groupby

NUM_FEATURES = 300


def simple_average_count():



def complex_average_count():
# Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count by one
    # Realizamos un conteo de los centroides que hay en el texto
    # Obtenemos a su vez el conteo mayor de centroides para hacer el corte mas adelante
    centroids_count = {}
    max_count = 0
    for word in wordlist.split():
        #print("word: ", word)
        if word in word_centroid_map:
            index = word_centroid_map[word]
            if index in centroids_count: 
              centroids_count[index] = centroids_count[index] + 1
              # Actualizamos el valor de maximo conteo si procede
              if centroids_count[index] > max_count:
                max_count = centroids_count[index]
            else:
              centroids_count[index] = 1
    
    
    
    #Filtramos aquellos valores del diccionario que tienen una ocurrencia mayor a la fijada
    if not max_df == 1:
      # Nos quedamos con los terminos menos frecuentes 
      max_count_threshold = math.ceil(max_count * max_df)
      #print(">> Max count of centroids obtained: ", max_count)
      #print(">> max_count_threshold: ", max_count_threshold)  
      #print(">> Count of different centroids (before filtering): ", len(centroids_count.items()))
      filtered_centroids = {k: v for k, v in centroids_count.items() if  v < max_count_threshold}
      #print(">> Count of different centroids (after filtering): ", len(centroids_count.items()))
    else:
      filtered_centroids = centroids_count
    
    
    # Obtenemos la media agregada de los valores restantes
    total_count = 0
    featureVec = np.zeros((NUM_FEATURES,), dtype="float32")
    for index, count in filtered_centroids.items():
    #for index, count in centroids_count.items():
      #print("Actual index: ",index, " center:" , centers[index])
      vec = np.multiply(centers[index], count) #Multiplicamos el vector por su ocurrencia
      featureVec = np.add(featureVec, vec)
      total_count = total_count + count
    
    #print(">> Total count of words: ", total_count)
    featureVec = np.divide(featureVec, total_count)
    #print (">> Aggregated Mean: ", featureVec)
    print("----------------------------------------")
    
# Con la clusterización asignamos un centroide a cada palabra, por lo que de esta forma podemos definir una funcion
# para convertir las noticias en una bolsa de centroides. 
# esta funcion devuelve un array numpy por cada review, cada una con tantas features como clusteres
def create_bag_of_centroids(wordlist, word_centroid_map, clusters, max_df=0.8):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    centers = np.array(clusters.cluster_centers_)
    #print("First centre: ", centers[0])
    
    if max_df == 1:
      simple_average_count()
    else:
      complex_average_count()
      
    #print("len(featureVec) ", len(featureVec))
    return np.array(featureVec)

def executeClusterization(word2vec_model, binary, classifier, cluster_size=600 ,train_data=None, test_data=None):
    basePath = "./fnc-1-original/aggregatedDatasets/"
    start = time.time() # Start time
    print("BINARY: ", binary)
    #model = KeyedVectors.load_word2vec_format(model_name)
    if binary:
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    else:
        model = gensim.models.Word2Vec.load(word2vec_model)

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0

    num_clusters = round(word_vectors.shape[0] / cluster_size)
    print("> Creando clusteres a partir del modelo cargado...")
    print("> Numero de terminos del modelo: ", word_vectors.shape[0])
    print("> Tamaño del cluster: ", cluster_size)
    print("> Numero de clusteres: ", num_clusters)

    # Inicializa un objeto de k-means y lo usa para extraer centroides
    n_jobs = multiprocessing.cpu_count()
    # kmeans_clustering = KMeans(n_clusters= num_clusters, max_iter=100, n_jobs=1)
    kmeans_clustering = MiniBatchKMeans(n_clusters= num_clusters, init_size=num_clusters*3)
    # En idx guardamos el cluster asociado a cada palabra
    # print("> Tiempo empleado en inicializar modelo Kmeans")
    print("> Inicio fit_predict...")
    idx = kmeans_clustering.fit_predict(word_vectors)
    print(idx)
    end = time.time()
    elapsed = end - start
    print("> Tiempo empleado en realizar el clustering de kmeans: ", elapsed, " seconds.")

    #Creamos un diccionario de word/index, que mapea cada palabra del vocabulario con su cluster asociado
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    #print(word_centroid_map)

    # Exploramos un poco los clusteres creados (los 10 primeros)
#    for cluster in range(0,10):
#        print("\nCluster ", cluster)
#        # Imprimimos todas las palabras del cluster
#        words = []
#        for i in range(0,len(word_centroid_map.values())):
#            if(list(word_centroid_map.values())[i] == cluster):
#                words.append(list(word_centroid_map.keys())[i])
#        print(words)
#        
    # Creamos el modelo de bag of centroids 
    # Reservamos un array para el conjunto de entrenamiento de bag of centroids (por razones de velocidad)

    train_formatting_start = time.time()
    trainDataPath = basePath + "train_data_aggregated_mini.csv"
    trainData = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)

    print(">> Generating mean of cluster centroids for training data...")
    # Transformamos el set de entrenamiento a bolsa de centroides
    # En los headlines no hacemos ningun filtrado en base a la ocurrencia (max_df = 1)
    train_headlines_vecs = Parallel(n_jobs=n_jobs, verbose= 10)(delayed(create_bag_of_centroids)(report, word_centroid_map, kmeans_clustering, 1) for report in trainData['Headline'])	
    train_articles_vecs = Parallel(n_jobs=n_jobs, verbose= 10)(delayed(create_bag_of_centroids)(report, word_centroid_map, kmeans_clustering) for report in trainData['ArticleBody'])	
    
    print(">> Composing train data input features...")
    print("Train Headlines: ", len(train_headlines_vecs))
    print("Train ArticleBodies: ", len(train_articles_vecs))
    train_centroids = []
    for headline,report in zip(train_headlines_vecs, train_articles_vecs):
        train_sample = np.append(headline, report)
        train_centroids.append(train_sample)
        #train_centroids = np.append(train_centroids,train_sample)

    train_formatting_end = time.time()


    testDataPath = basePath + "test_data_aggregated_mini.csv"
    testData = pd.read_csv(testDataPath, header=0, delimiter=",", quoting=1)
    #  Transformamos el set test a bolsa de centroids
    print(">> Generating mean of cluster centroids for testing data...")  
    # En los headlines no hacemos ningun filtrado en base a la ocurrencia (max_df = 1)
    test_headlines_vecs = Parallel(n_jobs=n_jobs, verbose= 10)(delayed(create_bag_of_centroids)(report, word_centroid_map, kmeans_clustering, 1) for report in testData['Headline'])	
    test_articles_vecs = Parallel(n_jobs=n_jobs, verbose= 10)(delayed(create_bag_of_centroids)(report, word_centroid_map, kmeans_clustering) for report in testData['ArticleBody'])	

    # Transformamos el set de entrenamiento a bolsa de centroides
    print(">> Composing test data input features...")
    test_centroids = []
    for headline,report in zip(test_headlines_vecs, test_articles_vecs):
        test_sample = np.append(headline, report)
        test_centroids.append(test_sample)
        #test_centroids = np.append(test_centroids, test_sample)

    test_formatting_end = time.time()
    print("> Tiempo empleado en formatear los datos de entrenamiento: ", train_formatting_end - train_formatting_start)
    print("> Tiempo empleado en formatear los datos de test: ", test_formatting_end - train_formatting_end)

    # Llamamos al clasificador con los datos compuestos
    classify_start = time.time()
    classification_results = {}
    
    if classifier == 'MLP':
        train_centroids = Imputer().fit_transform(train_centroids)
        test_centroids = Imputer().fit_transform(test_centroids)
        # Modelo basado en un MultiLayer Perceptron
        classification_results = textModelClassifier.modelClassifier(np.array(train_centroids), trainData['Stance'], np.array(test_centroids), testData['Stance'])
    elif classifier == 'RF':
        # Modelo basado en un randomForest sencillo
        train_centroids = Imputer().fit_transform(train_centroids)
        test_centroids = Imputer().fit_transform(test_centroids)
        classification_results = randomClassifier(np.array(train_centroids), trainData['Stance'], np.array(test_centroids), testData['Stance'])
    
    classify_end = time.time()
    print("> Tiempo empleado en ejecutar el clasificador: ", classify_end - classify_start)
    
    # Ponemos en un csv los tiempos de ejecucion para compararlos más adelante
    # Se genera un fichero por dia
    csvOutputDir = "./executionStats/"
    date = time.strftime("%Y-%m-%d")
    # validationDesc = "validation" if validation else ""
    output_file = csvOutputDir + executionDesc + "_execution_" + date + ".csv"
    fieldNames = ["date", "executionDesc", "textModelFeatures", "modelName", "loadModelTime", \
        "trainDataFormattingTime","trainDataFeatureVecsTime","testDataFormattingTime","testDataFeatureVecsTime", "totalExecutionTime",\
        "trainInstances", "testInstances", "modelTrained", "clusterSize","modelExecutionTime","trainAccuracy", "testAccuracy",\
        "confusionMatrix", "averagePrecision", "recall"]
    
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
        executionData = {
         "date": time.strftime("%Y-%m-%d %H:%M"),
         "executionDesc": executionDesc, 
         "textModelFeatures": np.array(trainDataInputs).shape[1], 
         "modelName": word2vec_model,
         "loadModelTime": round(loadModelTime,2),
         "trainDataFormattingTime": round(trainDataFormattingTime,2),
         "trainDataFeatureVecsTime": round(trainFeatureVecsTime,2),
         "testDataFormattingTime": round(testDataFormattingTime,2),
         "testDataFeatureVecsTime": round(testDataFeatureVecsTime,2),
         "totalExecutionTime": round(totalExecutionTime,2),
         "trainInstances": trainData.shape[0],
         "testInstances": testData.shape[0],
         "modelTrained": model_executed,
         "clusterSize": cluster_size,
         "modelExecutionTime": round(modelExecutionTime,2),
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
    # Cargamos el modelo
    word2vec_model = sys.argv[1]
    binary = sys.argv[2]
    classifier = sys.argv[3]
    print("BINARY: ", binary)
    executeClusterization(word2vec_model, binary, classifier)