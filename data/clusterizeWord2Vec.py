from sklearn.cluster import KMeans
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



start = time.time() # Start time

# Cargamos el modelo
model_name = sys.argv[1]

# model = KeyedVectors.load_word2vec_format(model_name)
model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.wv.syn0

num_clusters = round(word_vectors.shape[0] / 5)

# Inicializa un objeto de k-means y lo usa para extraer centroides
n_jobs = multiprocessing.cpu_count()
kmeans_clustering = KMeans(n_clusters= num_clusters, max_iter=100, n_jobs=n_jobs)
# En idx guardamos el cluster asociado a cada palabra
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()
elapsed = end - start
print("Time taken for K means clustering: ", elapsed, " seconds.")

#Creamos un diccionario de word/index, que mapea cada palabra del vocabulario con su cluster asociado
word_centroid_map = dict(zip(model.index2word, idx))


# Exploramos un poco los clusteres creados (los 10 primeros)
for cluster in xrange(0,10):
    print("\nCluster ", cluster)
    # Imprimimos todas las palabras del cluster
    words = []
    for i in xrange(0,len(word_centroid_map.values())):
        if(word_centroid_map.values()[i] == cluster):
            words.append(word_centroid_map.keys()[i])
    print(words)
    

# Creamos el modelo de bag of centroids 
# Reservamos un array para el conjunto de entrenamiento de bag of centroids (por razones de velocidad)
# 

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


# Utilizamos un modelo de random forest para ver como se comporta el modelo creado
# forest = RandomForestClassifier(n_estimators=100)

# print("Fitting a random fores to labeled training data...")
# forest = forest.fit(train_centroids, train["Stance"])
# result = forest.predict(test_centroids)

# # Escribimos los resultados
# output = pd.Dataframe(data={"id": test["id"], "sentiment": result})
# output.to_csv("BagOfcentroids.csv", index=False, quoting=3)

# Llamamos al clasificador con los datos compuestos
textModelClassifier.modelClassifier(train_centroids, trainData['Stance'], test_centroids, testData['Stance'])



