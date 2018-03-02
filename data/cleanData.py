#
# Script that cleans data_stances and data_bodies
print(">>> Importing dependencies...")
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
        #print word," ES UN VERBO --- lemmatize: ", WordNetLemmatizer.lemmatize(word,pos='v')
        return WordNetLemmatizer.lemmatize(word,pos='v')
    else:
        #print word," ES OTRA PALABRA"
        return WordNetLemmatizer.lemmatize(word)

# Body -> textTag: "articleBody"
#          outputFilePath:"./fnc-1-original/cleanDatasets/train_bodies_clean.csv"

def cleanTextData(stanceData,inputFilePath, outputFilePath, printLogs=False, cleanStopWords = True):
    # Read the dataset
    #INPUT_FILE = "train_bodies_example.csv"
    # INPUT_FILE = "train_bodies.csv"
    # basePath = "./fnc-1-original/"
    # outputDir = basePath + "cleanDatasets/"
    trainFile = pd.read_csv(inputFilePath,header=0,delimiter=",", quoting=1)
    # trainBodiesCleanPath= outputDir + "train_bodies_clean.csv"
    print(">>> Read file ", inputFilePath , "shape:", trainFile.shape)
  
    textTag = "Headline" if stanceData else "articleBody"         
    
    cleanedBodies = []

    # Open a new file to write the results
    with open(outputFilePath, 'w') as trainBodiesClean:
        fieldnames = ["Headline","Body ID","Stance"] if stanceData else ['Body ID', 'articleBody']
        writer = csv.DictWriter(trainBodiesClean, fieldnames=fieldnames)
        writer.writeheader()

        i = 0 
        print(">>> Processing Bodies...")

        for index,line in trainFile.iterrows():        
            # 1. Change all numbers by "NUM" tag and remove all puntuation symbols by a single space
            body = re.sub("[0-9]+", "NUM", line[textTag])
            body = re.sub("[^a-zA-Z]", " ", body)
            
            # 2. Convert to lower case all characters in body
            body = body.lower()
            
            # 3. TODO: Remove javascript code
            
            # 4. Tokenize body
            bodyWords = body.split()
            
            # 5. Remove stop-words from body, if specified
            if cleanStopWords:
                stopSet = set(stopwords.words("english"))
                bodyWords = [word for word in bodyWords if not word in stopSet]
            
            # 6. POS tagging and Lemmatize body
            posTagging = nltk.pos_tag(bodyWords)
            # bodyWords = list(map(WordNetLemmatizer.lemmatize,bodyWords))
            bodyLemmatized = []
            for taggedWord in posTagging:
                bodyLemmatized.append(lemmatizeWord(taggedWord))
            
            # Write to a file the processed line
            # We join again the body cleaned Word list
            # bodyClean = " ".join(bodyWords)
            bodyClean = " ".join(bodyLemmatized)
            cleanBodyLine = {"Body ID": line["Body ID"], textTag: bodyClean}
            if stanceData:
                cleanBodyLine["Stance"] = line["Stance"]
            
            
            writer.writerow(cleanBodyLine)

            # We log partially the process
            if i%1000== 0 and printLogs:
                print(">>>> Iteration ", i, " of ", trainFile.shape[0])
                print("0. Original Body: ")
                print(line[textTag])
                print("--------------------------------------------------")
                print(">>>> 1 - 5. Tokenized words and processed body ")
                print(bodyWords)
                print("--------------------------------------------------")
                print(">>> 6. BodyLemmatized: ")
                print(bodyLemmatized)
                print(">>>> Final cleaned body: ")
                print(str(cleanBodyLine))
                print("--------------------------------------------------")
                print("##################################################")

            #Append the processed Body to list of processed texts and increase iterations
            cleanedBodies.append(cleanBodyLine[textTag])
            i = i + 1 
    return cleanedBodies
    
    # TODO's
    # Dejar comentada la parte del codigo de stemming (para poder comparar mas tarde)
    # - Quitar codigo javascript presente en el cuerpo de las noticias
    # BF_STATIC.timequeue.push(function () { document.getElementById(""update_article_update_time_4050373"").innerHTML = UI.dateFormat.get_formatted_date('2014-10-17 18:18:33 -0400', 'update'); });" (EN PRINCIPIO PODRIA ELIMINARSE A MANO)

# if __name__ == "__main__":
#     cleanTextData()
