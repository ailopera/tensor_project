from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import sys
import gensim
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import textModelClassifier
import multiprocessing

# Con la clusterización asignamos un centroide a cada palabra, por lo que de esta forma podemos definir una funcion
# para convertir las noticias en una bolsa de centroides. 
# esta funcion devuelve un array numpy por cada review, cada una con tantas features como clusteres
def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clisters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1

    # Preallocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which clustter it belongs to, and increment that cluster count by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    
    # Return the "bag of centroids"
    return bag_of_centroids

def executeClusterization(word2vec_model, binary, classifier, cluster_size=200 ,train_data=None, test_data=None):
    basePath = "./fnc-1-original/aggregatedDatasets/"
    start = time.time() # Start time

    #model = KeyedVectors.load_word2vec_format(model_name)
    if binary == True:
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
    kmeans_clustering = MiniBatchKMeans(n_clusters= num_clusters)
    # En idx guardamos el cluster asociado a cada palabra
    # print("> Tiempo empleado en inicializar modelo Kmeans")
    print("> Inicio fit_predict...")
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print("> Tiempo empleado en realizar el clustering de kmeans: ", elapsed, " seconds.")

    #Creamos un diccionario de word/index, que mapea cada palabra del vocabulario con su cluster asociado
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # Exploramos un poco los clusteres creados (los 10 primeros)
    for cluster in range(0,10):
        print("\nCluster ", cluster)
        # Imprimimos todas las palabras del cluster
        words = []
        for i in range(0,len(word_centroid_map.values())):
            if(list(word_centroid_map.values())[i] == cluster):
                words.append(list(word_centroid_map.keys())[i])
        print(words)
        
    # Creamos el modelo de bag of centroids 
    # Reservamos un array para el conjunto de entrenamiento de bag of centroids (por razones de velocidad)

    train_formatting_start = time.time()
    trainDataPath = basePath + "train_data_aggregated.csv"
    trainData = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)

    train_centroids = np.zeros((trainData.shape[0], num_clusters), dtype="float32")
    print(">> Generating bag of centroids for training data...")
    # Transformamos el set de entrenamiento a bolsa de centroides
    for report in trainData:
        train_articleBody = create_bag_of_centroids(report['ArticleBody'], word_centroid_map)
        train_body = create_bag_of_centroids(report['Headline'], word_centroid_map)
        train_sample = np.append(train_articleBody, train_body)
        train_centroids.append(train_sample)

    train_formatting_end = time.time()


    testDataPath = basePath + "test_data_aggregated.csv"
    testData = pd.read_csv(testDataPath, header=0, delimiter=",", quoting=1)
    #  Transformamos el set test a bolsa de centroids
    test_centroids = np.zeros((testData.shape, num_clusters), dtype="float32")
    print(">> Generating bag of centroids for testing data...")
    # Transformamos el set de entrenamiento a bolsa de centroides
    for report in testData:
        test_articleBody = create_bag_of_centroids(report['ArticleBody'], word_centroid_map)
        test_body = create_bag_of_centroids(report['Headline'], word_centroid_map)
        test_sample = np.append(test_articleBody, test_body)
        test_centroids.append(test_sample)

    test_formatting_end = time.time()
    print("> Tiempo empleado en formatear los datos de entrenamiento: ", train_formatting_end - train_formatting_start)
    print("> Tiempo empleado en formatear los datos de test: ", test_formatting_end - train_formatting_end)

    # Llamamos al clasificador con los datos compuestos
    classify_start = time.time()
    textModelClassifier.modelClassifier(train_centroids, trainData['Stance'], test_centroids, testData['Stance'])
    classify_end = time.time()
    print("> Tiempo empleado en ejecutar el clasificador: ", classify_end - classify_start)


if __name__ == "__main__":
    # Cargamos el modelo
    word2vec_model = sys.argv[1]
    binary = sys.argv[2]
    classifier = sys.argv[3]
    executeClusterization(word2vec_model, binary, classifier)