# Script that cleans data_stances and data_bodies
import pandas as pd
import re
import nltk
nltk.download()
from nltk.corpus import stopwords
import csv
# Read the dataset
basePath = "./fnc-1-original/"
train_stances = pd.read_csv(basePath + "train_stances_example.csv",header=0,delimiter=",", quoting=1)
trainStancesPath = "train_stances_clean.csv"
print ">>> Read file train_stances.csv, shape:", train_stances.shape
print train_stances["Headline"][0]


# train_body = pd.read_csv(basePath + "train_bodies.csv",header=0,delimiter=",", quoting=1)
# print(">>> Read file train_bodies.csv, shape:", train_body.shape )

# Open a new file to write the results
with open(trainStancesPath, 'w') as trainStancesClean:
    fieldnames = ['Headline', 'Body_ID' 'Stance']
    writer = csv.DictWriter(trainStancesClean, fieldnames=fieldnames)
    writer.writeheader()
    for line in train_stances:
	    # 1. Change al numbers by "NUM" tag and remove all puntuation symbols by a single space
	    headline = re.sub("[1-9]*", "NUM",line["Headline"])
	    headline = re.sub("[^a-zA-Z]", " ",line["Headline"])
	    # 2. Convert to lower case all characters in headline
	    headline = headline.lower()
	    # 3. Tokenize headline
	    headlineWords = headline.split()
	    # 4. Remove stop-words from headline
	    headlineWords = [word for word in headlineWords if not word in stopwords.words("english")]
	    print ">>>> Headline words ", headlineWords

	    # Write to a file the processed line
	    cleanLine = {"Headline": headlineWords,"Body_ID": line["Body ID"],"Stance": line["Stance"]}
	    writer.writerow(cleanLine)
	    print "processed line: ", str(cleanLine)
# TODO's
# - Hacemos lemmatizing: ponemos en su forma base las palabras
# - Quitar codigo javascript presente en el cuerpo de las noticias
# BF_STATIC.timequeue.push(function () { document.getElementById(""update_article_update_time_4050373"").innerHTML = UI.dateFormat.get_formatted_date('2014-10-17 18:18:33 -0400', 'update'); });" (EN PRINCIPIO PODRIA ELIMINARSE A MANO)
