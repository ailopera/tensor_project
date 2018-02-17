import pandas as pd
import ntlk.data
import logging
from gensim.models import word2vec
from bs4 import BeautifulSoup

#Script que genera el modelo de word2Vec


def news_to_wordlist(news, remove_stopwords=False):
    # This time we won't remove the stopwords and numbers

    # 0. Remove HTML tags
    body = BeatifulSoup(news).get_text() 
    # 1. Change all numbers by "NUM" tag and remove all puntuation symbols by a single space
    # body = re.sub("[0-9]+", "NUM", news)
    # body = re.sub("[^a-zA-Z]", " ", body)
    
    # 2. Convert to lower case all characters in body
    body = body.lower()
    
    # 3. TODO: Remove javascript code & URLS
    
    # 4. Tokenize body
    bodyWords = body.split()
    
    # 5. Remove stop-words from body
    # if remove_stopwords:
        # stopSet = set(stopwords.words("english"))
        # bodyWords = [word for word in bodyWords if not word in stopSet]
        
    # # 6. POS tagging and Lemmatize body
    # posTagging = nltk.pos_tag(bodyWords)
    # # bodyWords = list(map(WordNetLemmatizer.lemmatize,bodyWords))
    # bodyLemmatized = []
    # for taggedWord in posTagging:
    #     bodyLemmatized.append(lemmatizeWord(taggedWord))
    
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

def trainWord2Vec(sentences):
    logging.basicConfig(format='%s(asctime)s: %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40 # Minimum word count
    num_workers = 4 # Number of threads to run in parallel
    context = 10 # Context window size
    downsampling = 1e-3 #Downsample setting for frequent words 

    # Initialize and train the model (this will take some time)
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, \
    window = context, sample=downsampling)

    # If you don't plan to train the model any further, calling init_sims 
    # will make the model much more memory-efficient
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)


def makeWord2VecModel(textTag):
    basePath = "./fnc-1-original/"
    outputDir = basePath + "cleanDatasets/"
    inputFilePath = basePath + "train_bodies.csv" 
    inputUnlabeledFile = basePath + "test_stances_unlabeled.csv"
    # Leemos los ficheros etiquetados y sin etiquetar
    trainFile = pd.read_csv(inputFilePath,header=0,delimiter=",", quoting=1)
    print(">>> Read file ", inputFilePath , "shape:", trainFile.shape)
    # Si tuvieramos datos sin etiquetar podriamos utilizarlos igualmente en el entrenamiento
    # ya que word2vec no requiere de datos etiquetados
    # unlabeled_train = pd.read_csv(inputUnlabeledFile, header=0,delimiter=",", quoting=1)
    # print(">>> Read file ", inputUnlabeledFile , "shape:", unlabeled_train.shape)

    # Word2Vec se espera un formato de lista de listas (lista de frases, cada frase una lista de palabras),
    # asi que procesamos el texto para que tenga ese formato 
    # Utilizaremos el punkTokenizer de nltk

    #Download the puntk tokenizer for sentence splitting
    ntlk.download()

    # Load the puntk tokenizer
    tokenizer = ntlk.data.load('tokenizer/punkt/english.pickle')

    sentences = []
    print("> Parsing sentences from training set")

    for index,line in trainFile.iterrows():
        sentences += news_to_sentences(line[textTag], tokenizer)
    
    # Check how many sentences we have in total
    print("> #Sentences: ", len(sentences))

    print ("> First Sentence : ", sentences[0])
    print ("> Second Sentence : ", sentences[1])
    trainWord2Vec(sentences)
