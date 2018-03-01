from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0

num_clusters = word_vectors.shape[0] / 5

# Inicializa un objeto de k-means y lo usa para extraer centroides
kmeans_clustering = KMeans(n_clusters= num_clusters)
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
    print words
    
