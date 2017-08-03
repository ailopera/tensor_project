# Script that cleans data_stances and data_bodies
print ">>> Importing dependencies..."
import pandas as pd
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
WordNetLemmatizer = WordNetLemmatizer()

# Read the dataset
basePath = "./fnc-1-original/"
train_stances = pd.read_csv(basePath + "train_stances_example.csv",header=0,delimiter=",", quoting=1)
trainStancesPath = "train_stances_clean.csv"
print ">>> Read file train_stances.csv, shape:", train_stances.shape
print train_stances
print type(train_stances)

# train_body = pd.read_csv(basePath + "train_bodies.csv",header=0,delimiter=",", quoting=1)
# print(">>> Read file train_bodies.csv, shape:", train_body.shape )

# Open a new file to write the results
with open(trainStancesPath, 'w') as trainStancesClean:
    fieldnames = ['Headline', 'Body ID', 'Stance']
    writer = csv.DictWriter(trainStancesClean, fieldnames=fieldnames)
    writer.writeheader()
    # for line in train_stances:
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
        # 5. Lemmatize headline
        headlineWords = list(map(WordNetLemmatizer.lemmatize,headlineWords))
        
        print ">>>> Headline words ", headlineWords

        # Write to a file the processed line
        # We join again the headline cleaned Word list
        headlineClean = " ".join(headlineWords)
        cleanLine = {"Headline": headlineClean,"Body ID": line["Body ID"],"Stance": line["Stance"]}
        writer.writerow(cleanLine)
        print "processed line: ", str(cleanLine)


# TODO's
#Hay que hacer un POS taggging de las palabras para que se haga bien el lemmatizer
# Dejar comentada la parte del codigo de stemming (para poder comparar más tarde)
# - Quitar codigo javascript presente en el cuerpo de las noticias
# BF_STATIC.timequeue.push(function () { document.getElementById(""update_article_update_time_4050373"").innerHTML = UI.dateFormat.get_formatted_date('2014-10-17 18:18:33 -0400', 'update'); });" (EN PRINCIPIO PODRIA ELIMINARSE A MANO)
