import pandas as pd
import re, os, csv, time
import nltk.data
import logging
#from gensim import models
from gensim.models import word2vec
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
WordNetLemmatizer = WordNetLemmatizer()

# from nltk import word_tokenize, sent_tokenize
# nltk.download()
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
#Script que genera el modelo de word2Vec



# AUXILIAR FUNCTIONS #
# Determines whether a tag belongs to a verb
def isVerbTag(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# Does the lemmatizing of a specific pos recognition (word, tag)
# NOTA: Tenemos que hacer un postagging previo, ya que necesitamos "ayudar" al lemmatizer
# y decirle aquellas palabras que son verbos
def lemmatizeWord(posTag):
    word = posTag[0]
    tag = posTag[1]
    # print "word: ", word, " tag: ", tag
    if isVerbTag(tag):
        #print word," ES UN VERBO --- lemmatize: ", WordNetLemmatizer.lemmatize(word,pos='v')
        return WordNetLemmatizer.lemmatize(word,pos='v')
    else:
        #print word," ES OTRA PALABRA"
        return WordNetLemmatizer.lemmatize(word)


# Si no se especifica que limpie el texto, simplemente spliteamos el texto
def news_to_wordlist(news, remove_stopwords=False,clean_text=True):
    # This time we won't remove the stopwords and numbers
    if clean_text:
        # 0. Remove HTML tags
        body = BeautifulSoup(news,"html.parser").get_text() 
        # 1. Change all numbers by "NUM" tag and remove all puntuation symbols by a single space
        body = re.sub("[0-9]+", "NUM", news)
        body = re.sub("[^a-zA-Z]", " ", body)
        
        # 2. Convert to lower case all characters in body
        body = body.lower()
        
        # 3. TODO: Remove javascript code & URLS
        body = re.sub('https?:\/\/.*[\r\n]*', " ", body)
    else:
        body = news


    # 4. Tokenize body
    bodyWords = body.split()
    
    # 5. Remove stop-words from body
    if remove_stopwords and clean_text:
        stopSet = set(stopwords.words("english"))
        bodyWords = [word for word in bodyWords if not word in stopSet]
        
    if clean_text:
        # # 6. POS tagging and Lemmatize body
        posTagging = nltk.pos_tag(bodyWords)
        bodyWords = list(map(WordNetLemmatizer.lemmatize,bodyWords))
        bodyLemmatized = []
        for taggedWord in posTagging:
            bodyLemmatized.append(lemmatizeWord(taggedWord))
        bodyWords = bodyLemmatized

    #Returns a list of words
    return(bodyWords)


#Function to split a news piece int parsed sentences. Returns a 
# list of sentences, where each sentence is a list of words
def news_to_sentences(news, tokenizer, remove_stopwords=False):
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(news.strip())

    #Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call news_to_wordlist to get a list of words
            sentences.append(news_to_wordlist(raw_sentence,remove_stopwords))

    #Return the list of sentences (each sentence is a list of words)
    # So this returns a list of lists
    return sentences

def trainWord2Vec(sentences,archiveTag):
    logging.basicConfig(format='%s(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 15 # Minimum word count
    num_workers = 4 # Number of threads to run in parallel
    context = 35 # Context window size
    downsampling = 1e-3 #Downsample setting for frequent words 

    # Initialize and train the model (this will take some time)
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, \
    window = context, sample=downsampling)

    # If you don't plan to train the model any further, calling init_sims 
    # will make the model much more memory-efficient
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = str(num_features) + "features_" + str(min_word_count) + "minwords_" + str(context) + "context" + archiveTag
    model.save(model_name)
    #models.Word2Vec.save_word2vec_format(model_name)
    return model_name

# def makeWord2VecModel(trainStance):
def makeWord2VecModel():
    basePath = "./fnc-1-original/"
    outputDir = basePath + "cleanDatasets/"
    # if trainStance:
    #     inputFilePath = basePath + "train_stances.csv" 
    #     fileTag = "STANCES"
    # else:
    #     inputFilePath = basePath + "train_bodies.csv" 
        # fileTag = "BODIES"
    
    # stancesFilePath = basePath + "train_stances.csv"
    # bodiesFilePath = basePath + "train_bodies.csv"

    fileTag = "ALL"

    # textTag = 'articleBody' if trainStance==False else 'Headline'
    # # Leemos los ficheros etiquetados y sin etiquetar
    # bodiesTrainFile = pd.read_csv(bodiesFilePath,header=0,delimiter=",", quoting=1)
    # stancesTrainFile = pd.read_csv(stancesFilePath,header=0,delimiter=",", quoting=1)
    # print(">>> Read file ", bodiesFilePath , "shape:", bodiesTrainFile.shape)
    # print(">>> Read file ", stancesFilePath , "shape:", stancesTrainFile.shape)
    
    aggregated_train_path = "./fnc-1-original/finalDatasets/train_partition.csv"
    aggregated_train_file = pd.read_csv(aggregated_train_path,header=0,delimiter=",", quoting=1)
    print(">>> Read file ", aggregated_train_file , "shape:", aggregated_train_file.shape)

    # Si tuvieramos datos sin etiquetar podriamos utilizarlos igualmente en el entrenamiento
    # ya que word2vec no requiere de datos etiquetados
    # inputUnlabeledFile = basePath + "test_stances_unlabeled.csv"
    # unlabeled_train = pd.read_csv(inputUnlabeledFile, header=0,delimiter=",", quoting=1)
    # print(">>> Read file ", inputUnlabeledFile , "shape:", unlabeled_train.shape)

    # Word2Vec se espera un formato de lista de listas (lista de frases, cada frase una lista de palabras),
    # asi que procesamos el texto para que tenga ese formato 
    # Utilizaremos el punkTokenizer de nltk

    #Download the puntk tokenizer for sentence splitting
    #nltk.download('puntk')

    # Load the puntk tokenizer
    tokenizer = nltk.data.load('tokenizer/punkt/english.pickle')

    sentences = []
    print("> Parsing sentences from training set")

    # Recorremos el fichero de titulares y cuerpos de noticias para crear el modelo
    # for index,line in bodiesTrainFile.iterrows():
    #     sentences += news_to_sentences(line['articleBody'], tokenizer)
    
    # for index,line in stancesTrainFile.iterrows():
    #     sentences += news_to_sentences(line['Headline'], tokenizer)
    
    start = time.time()
    for index,line in aggregated_train_file.iterrows():
        sentences += news_to_sentences(line['ArticleBody'], tokenizer)
        sentences += news_to_sentences(line['Headline'], tokenizer)
    end = time.time()
    formatTime = end - start

    # Check how many sentences we have in total
    print("> #Sentences: ", len(sentences))
    # print ("> First Sentence : ", sentences[0])
    # print ("> Second Sentence : ", sentences[1])
    
    start = time.time()
    model_name = trainWord2Vec(sentences, fileTag)
    end = time.time()
    trainTime = end - start

    # Export data to a csv file
    csvOutputDir = "./executionStats/"
    date = time.strftime("%Y-%m-%d")
    output_file = csvOutputDir + "word2vec_execution_" + date + ".csv"
    fieldNames = ["date", "modelName", "formatTime", "trainTime"]
    
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
        executionData = {
         "date": time.strftime("%Y-%m-%d %H:%M"),
         "modelName": model_name,
         "formatTime": formatTime,
         "trainTime": trainTime
         }
         
        newFile = os.stat(output_file).st_size == 0
        if newFile:
            writer.writeheader()
        writer.writerow(executionData)

        print(">> Stats exported to: ", output_file)


if __name__ == "__main__":
  #  makeWord2VecModel(False) # Para entrenar el fichero con los cuerpos de noticias
   makeWord2VecModel()
