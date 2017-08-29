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
train_stances = pd.read_csv(basePath + "train_stances_example.csv",header=0,delimiter=",", quoting=1)
trainStancesPath = outputDir + "train_stances_clean.csv"
print ">>> Read file train_stances.csv, shape:", train_stances.shape
print train_stances
print type(train_stances)


# Open a new file to write the results
with open(trainStancesPath, 'wb') as trainStancesClean:
    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(trainStancesClean, fieldnames=fieldnames)
    writer.writeheader()
    # for line in train_stances:
    i = 0 
    for index,line in train_stances.iterrows():
        print "--------------"
        print "line: ", line
        # 1. Change al numbers by "NUM" tag and remove all puntuation symbols by a single space
        headline = re.sub("[1-9]*", "NUM",line["Headline"])
        headline = re.sub("[^a-zA-Z]", " ",line["Headline"])
        # 2. Convert to lower case all characters in headline
        headline = headline.lower()
        # 3. Tokenize headline
        headlineWords = headline.split()
        # 4. Remove stop-words from headline
        stopSet = set(stopwords.words("english"))
        headlineWords = [word for word in headlineWords if not word in stopSet]
        # 5. POS tagging and Lemmatize headline
        posTagging = nltk.pos_tag(headlineWords)
        # headlineWords = list(map(WordNetLemmatizer.lemmatize,headlineWords))
        headLineLemmatized = []
        for taggedWord in posTagging:
            headLineLemmatized.append(lemmatizeWord(taggedWord))
        

        # Write to a file the processed line
        # We join again the headline cleaned Word list
        # headlineClean = " ".join(headLineWords)
        headlineClean = " ".join(headLineLemmatized)
        cleanLine = {"Headline": headlineClean,"Body ID": line["Body ID"],"Stance": line["Stance"]}
        writer.writerow(cleanLine)

        # We log partially the process
        if i%100== 0:
            print ">>>> Iteration ", i, " of ", train_stances.shape[1]
            print ">>>> Processed line: ", str(cleanLine)
            print ">>>> Headline words ", headlineWords
            print "--------------------------------------------------"

        i = i + 1 

# TODO's
# Dejar comentada la parte del codigo de stemming (para poder comparar mas tarde)
# - Quitar codigo javascript presente en el cuerpo de las noticias
# BF_STATIC.timequeue.push(function () { document.getElementById(""update_article_update_time_4050373"").innerHTML = UI.dateFormat.get_formatted_date('2014-10-17 18:18:33 -0400', 'update'); });" (EN PRINCIPIO PODRIA ELIMINARSE A MANO)

