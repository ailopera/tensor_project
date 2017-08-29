# Script that cleans data_stances and data_bodies
print ">>> Importing dependencies..."
import pandas as pd
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
WordNetLemmatizer = WordNetLemmatizer()

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
        print word," ES UN VERBO --- lemmatize: ", WordNetLemmatizer.lemmatize(word,pos='v')
        return WordNetLemmatizer.lemmatize(word,pos='v')
    else:
        print word," ES OTRA PALABRA"
        return WordNetLemmatizer.lemmatize(word)



# Read the dataset
basePath = "./fnc-1-original/"
outputDir = basePath + "cleanDatasets/"
train_stances = pd.read_csv(basePath + "train_bodies_example.csv",header=0,delimiter=",", quoting=1)
trainBodiesPath= outputDir + "train_bodies_clean.csv"
print ">>> Read file train_bodies.csv, shape:", train_stances.shape
print train_stances
print type(train_stances)


# Open a new file to write the results
with open(trainBodiesPath, 'wb') as trainStancesClean:
    fieldnames = ['Body ID', 'articleBody']
    writer = csv.DictWriter(trainStancesClean, fieldnames=fieldnames)
    writer.writeheader()
    # for line in train_stances:
    i = 0 
    for index,line in train_stances.iterrows():
        print "--------------"
        print "line: ", line
        
        # 1. Change al numbers by "NUM" tag and remove all puntuation symbols by a single space
        body = re.sub("[1-9]*", "NUM",line["articleBody"])
        body = re.sub("[^a-zA-Z]", " ",line["articleBody"])
        
        # 2. Convert to lower case all characters in body
        body = body.lower()
        
        # 3. TODO: Remove javascript code
        # 3. Tokenize body
        bodyWords = body.split()
        
        # 4. Remove stop-words from body
        stopSet = set(stopwords.words("english"))
        bodyWords = [word for word in bodyWords if not word in stopSet]
        
        # 5. POS tagging and Lemmatize body
        posTagging = nltk.pos_tag(bodyWords)
        # bodyWords = list(map(WordNetLemmatizer.lemmatize,bodyWords))
        bodyLemmatized = []
        for taggedWord in posTagging:
            bodyLemmatized.append(lemmatizeWord(taggedWord))
        

        # Write to a file the processed line
        # We join again the body cleaned Word list
        # bodyClean = " ".join(bodyWords)
        bodyClean = " ".join(bodyLemmatized)
        cleanLine = {"Body ID": line["Body ID"], "articleBody": bodyClean}
        writer.writerow(cleanLine)

        # We log partially the process
        if i%100== 0:
            print ">>>> Iteration ", i, " of ", train_stances.shape[1]
            print ">>>> Processed line: ", str(cleanLine)
            print ">>>> body words ", bodyWords
            print "--------------------------------------------------"

        i = i + 1 

# TODO's
# Dejar comentada la parte del codigo de stemming (para poder comparar mas tarde)
# - Quitar codigo javascript presente en el cuerpo de las noticias
# BF_STATIC.timequeue.push(function () { document.getElementById(""update_article_update_time_4050373"").innerHTML = UI.dateFormat.get_formatted_date('2014-10-17 18:18:33 -0400', 'update'); });" (EN PRINCIPIO PODRIA ELIMINARSE A MANO)

