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
#import textModelClassifier # Primer modelo de clasificador basico, no parammetrizable
# import textModelClassifierParametrized # modelo de clasificador paramatretrizado
import recurrentClassifier
from randomForestClassifier import randomClassifier
from imblearn.over_sampling import SMOTE
LOG_ENABLED = False

# Check-list
# Cargamos los datasets

# Con el tokenizador pntk de NLTK detectamos las frases

# Calculamos la longitud maxima de frase obtenida y la exportamos a un csv

# Limitamos la longitud de frase a un valor maximo


# Creamos el vector de características para el titular y para el cuerpo de noticias
# Conjunto de Entrenamiento

# Conjunto de test


# Aplicamos SMOTE para contrarrestar el desbalanceo de muestras

def writeTextStats(cleaned_texts, label="articleBody"):
    output_file = "stats/trainDataStats_" + label +".csv"
    with open(output_file, 'a') s csv_file:
        header = ["SentenceCount", "MaxSentenceLength"]
        writer = csv.DictWriter(csv_file, fieldnames = header)
        newFile = os.stat(output_file).st_size == 0 
        if newFile:
            print("Writting Training stats for ", label)
            writer.writeheader()
            for text in cleaned_texts:
                max_sentence_len = len(max(text, key=len)) 
                row = { "SentenceCount": len(text)
                    "MaxSentenceLength": max_sentence_len }
                writer.writerow(row)

def makeFeatureVec(words, model, num_features, index2word_set, log=False):
    #Function to average all of the word vectors in a given paragraph
    #Pre-initialize an empty numpy array (for speed)
    # featureVec = np.zeros((num_features,), dtype="float32")
    featureVec = []
    nwords = 0.
    
    # Obtenemos una lista de listas que representan todas las frases texto
    sentences_list = word2VecModel.news_to_sentences(words)
    #Loop over each word in the review and, if it is in the model's vocabulary, 
    # add its feature vector to the total
    feature_sentence = []
    for sentence in sentences_list:
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1
                feature_sentence = np.append(feature_sentence, model[word])
        featureVec = np.append(featureVec, feature_sentence)
        feature_sentence = []
    
    return featureVec


def getFeatureVecs(news, model, num_features):
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
        
        
        newsFeatureVecs[counter] = makeFeatureVec(report, model, num_features, index2word_set)
        counter = counter + 1

    # Version paralela del computo de vectores de caracteristicas
    # num_cores = multiprocessing.cpu_count()
    # newsFeatureVecs = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeFeatureVec)(report, model, num_features, index2word_set) for report in news)

    return newsFeatureVecs	



def makeWordList(text):
    wordList = word2VecModel.news_to_wordlist(text,remove_stopwords=True, clean_text=False)
    return wordList

def executeVectorFeaturing(word2vec_model, model_executed, binary, train_data=None, test_data=None, validation=False, smote="", classifier_config=None):
    # basePath = "./fnc-1-original/aggregatedDatasets/"
    num_features = 300
    # executionDesc = "vector_Emmbeddings"
    
    start = time.time()
    execution_start = start
    print("> Word2vec_model:", word2vec_model)
    print("> Binary: ", binary)
    
    if binary == True:
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    else:
        model = gensim.models.Word2Vec.load(word2vec_model)
        
    end = time.time()
    loadModelTime = end - start
    print("> Tiempo empleado en cargar el modelo: ", loadModelTime)
    

    # Generamos representaciones de vectores de medias
    # En este caso si quitamos las stopwords, a diferencia a cuando creamos el modelo
    # Las stopwords pueden introducir ruido en el calculo de los vectores de medias
    
    print(">> Generating word2vec input model and getting embeddings for train data...")
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
    
    # Escribimos en un csv la longitud de cada frase y cuerpo de noticia
    writeTextStats(clean_train_headlines, "headline")
    writeTextStats(clean_train_articleBodies)

    # print(">> Getting feature vectors for train headlines...")
    # start = time.time()
    # trainDataVecsHeadline = getFeatureVecs(clean_train_headlines, model, num_features)
    # print(">> Getting feature vectors for train articleBodies...")
    # trainDataVecsArticleBody = getFeatureVecs(clean_train_articleBodies, model, num_features)
    # end = time.time()
    # trainFeatureVecsTime = end - start
    # print(">> Time spent on getting feature vectors for training data: ", trainFeatureVecsTime)
    
    # # Hacemos un append del vector de headline y el de la noticia
    # trainDataInputs = []
    # for sample in zip(trainDataVecsHeadline, trainDataVecsArticleBody):
    #     trainSample = np.append(sample[0],sample[1])
    #     trainDataInputs.append(trainSample)
    
    # # Hacemos lo mismo con los datos de test
    # print(">> Generating word2vec input model and applying embedding represenntation for test data...")
    # # testDataPath = basePath + "test_data_aggregated_mini.csv"

    # clean_test_articleBodies = []
    # clean_test_headlines = []
    # testDataVecs = {}

    # start = time.time()
    # clean_test_headlines = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['Headline'])
    # clean_test_articleBodies = Parallel(n_jobs=num_cores, verbose= 10)(delayed(makeWordList)(line) for line in testData['ArticleBody'])
    # end = time.time()
    # testDataFormattingTime = end - start
    # print(">> Time spent on formatting testing data: ", testDataFormattingTime)


    # print(">> Getting feature vectors for test articleBodies...")
    # start = time.time()
    # testDataVecsArticleBody = getFeatureVecs(clean_test_articleBodies, model, num_features)
    # print(">> Getting feature vectors for test headlines...")
    # testDataVecsHeadline = getFeatureVecs(clean_test_headlines, model, num_features)
    # end = time.time()
    # testDataFeatureVecsTime = end - start
    # print(">> Time spent on getting feature vectors for training data...", testDataFeatureVecsTime)
    
    # # Creamos un vector de 1x600 que contiene el titular y el cuerpo de noticia asociado, para alimentar el modelo de Machine Learning
    # testDataInputs = []
    # for sample in zip(testDataVecsHeadline, testDataVecsArticleBody):
    #     testSample = np.append(sample[0],sample[1])
    #     testDataInputs.append(testSample)

    # print("> Tamaño de los datos de entrada (entrenamiento): ", trainData.shape)
    # print("> Tamaño de los datos de entrada (test): ", testData.shape)

    

    # #Inferimos las muestras erroneas
    # trainDataInputs = Imputer().fit_transform(trainDataInputs)
    # testDataInputs = Imputer().fit_transform(testDataInputs)
    # #Aplicamos SMOTE si procede
    # if not smote == "":
    #     print(">> Applying SMOTE")
    #     trainDataInputs, train_labels = SMOTE(ratio=smote,random_state=None, n_jobs=4).fit_sample(trainDataInputs, trainData['Stance'])
    #     #testDataInputs, test_labels = SMOTE(ratio=smote,random_state=None, n_jobs=4).fit_sample(testDataInputs, testData['Stance'])
    # else:
    #     train_labels = trainData['Stance']

    
    # Llamamos al clasificador con los datos compuestos
    # start = time.time()
    # classification_results = {}
    # if model_executed == 'MLP':
    #     #Modelo basado en red neuronal recurrente
    #     classification_results = recurrentClassifier.modelClassifier(np.array(trainDataInputs), train_labels, np.array(testDataInputs), testData['Stance'], classifier_config)
    # elif model_executed == 'RF':
    #     # Modelo basado en un randomForest sencillo
    #     classification_results = randomClassifier(np.array(trainDataInputs), train_labels, np.array(testDataInputs), testData['Stance'])
    # else:
    #     print(">>> ERROR: No se ha ejecutado ningún modelo")

    # end = time.time()
    # modelExecutionTime = end - start
    # execution_end = end
    # totalExecutionTime = execution_end - execution_start
    # print("> Time spent on fiting and predicting: ", modelExecutionTime)
    # print(">> Metrics: ", classification_results)

    
    