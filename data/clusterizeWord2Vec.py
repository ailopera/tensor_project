from sklearn.cluster import KMeans, MiniBatchKMeans
import time
import sys
import gensim
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import textModelClassifier
import multiprocessing
NUM_FEATURES = 300
# Con la clusterización asignamos un centroide a cada palabra, por lo que de esta forma podemos definir una funcion
# para convertir las noticias en una bolsa de centroides. 
# esta funcion devuelve un array numpy por cada review, cada una con tantas features como clusteres
def create_bag_of_centroids(wordlist, word_centroid_map, clusters):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    centers = np.array(clusters.cluster_centers_)
    #print("First centre: ", centers[0])
    # Preallocate the bag of centroids vector (for speed)
    # bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which clustter it belongs to, and increment that cluster count by one
    bag_of_centroids = []
    for word in wordlist.split():
        #print("word: ", word)
        if word in word_centroid_map:
            index = word_centroid_map[word]
            #print("Index: ", index)
            #bag_of_centroids[index] += 1
            bag_of_centroids.append(centers[index])
    
    # Return the "bag of centroids"
    #print("len(bag_of_centroids) - BEFORE MEAN", len(bag_of_centroids))
    #print("bag_of_centroids ", bag_of_centroids)
    
    #Obtenemos la media en base a los centroides
    featureVec = np.zeros((NUM_FEATURES,), dtype="float32")
    nwords = 0
    for vec in bag_of_centroids:  
      nwords = nwords + 1
      featureVec = np.add(featureVec, vec)
    
    featureVec = np.divide(featureVec, nwords)
    #print("len(featureVec) ", len(featureVec))
    #print("FeatureVec ", featureVec)
    #time.sleep(20)
    return np.array(featureVec)

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
    trainDataPath = basePath + "train_data_aggregated.csv"
    trainData = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)

    train_centroids = np.zeros((trainData.shape[0]*2, num_clusters), dtype="float32")
    print(">> Generating mean of cluster centroids for training data...")
    # Transformamos el set de entrenamiento a bolsa de centroides
    for headline,report in zip(trainData['Headline'], trainData['ArticleBody']):
        train_articleBody = create_bag_of_centroids(report, word_centroid_map, kmeans_clustering)
        #print("Train_articleBody_features", len(test_articleBody))
        #print("Train_articleBody_features", test_articleBody)
        train_headline = create_bag_of_centroids(headline, word_centroid_map, kmeans_clustering)
        train_sample = np.append(train_headline, train_articleBody)
        train_centroids = np.append(train_centroids,train_sample)

    train_formatting_end = time.time()


    testDataPath = basePath + "test_data_aggregated.csv"
    testData = pd.read_csv(testDataPath, header=0, delimiter=",", quoting=1)
    #  Transformamos el set test a bolsa de centroids
    test_centroids = np.zeros((testData.shape, num_clusters), dtype="float32")
    print(">> Generating mean of cluster centroids for testing data...")
    # Transformamos el set de entrenamiento a bolsa de centroides
    for headline,report in zip(testData['Headline'], testData['ArticleBody']):
        test_articleBody = create_bag_of_centroids(report, word_centroid_map, kmeans_clustering)
        test_headline = create_bag_of_centroids(headline, word_centroid_map, kmeans_clustering)
        test_sample = np.append(test_headline, test_articleBody)
        #test_centroids.append(test_sample)
        test_centroids = np.append(test_centroids, test_sample)


    test_formatting_end = time.time()
    print("> Tiempo empleado en formatear los datos de entrenamiento: ", train_formatting_end - train_formatting_start)
    print("> Tiempo empleado en formatear los datos de test: ", test_formatting_end - train_formatting_end)

    # Llamamos al clasificador con los datos compuestos
    classify_start = time.time()
    classification_results = {}
    if classifier == 'RF':
        classification_results = textModelClassifier.modelClassifier(np.array(train_centroids), trainData['Stance'], np.array(test_centroids), testData['Stance'])
    elif classifier == 'MLP':
        classification_results = textModelClassifier.modelClassifier(np.array(test_centroids), trainData['Stance'], np.array(test_centroids), testData['Stance'])
    classify_end = time.time()
    print("> Tiempo empleado en ejecutar el clasificador: ", classify_end - classify_start)


if __name__ == "__main__":
    # Cargamos el modelo
    word2vec_model = sys.argv[1]
    binary = sys.argv[2]
    classifier = sys.argv[3]
    executeClusterization(word2vec_model, binary, classifier)