print("Training the random forest...")

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cleanData
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import createBOW

# Paso 1: Creamos el modelo de Bag of words de las reviews
inputStancesPath = "./fnc-1-original/train_stances.csv"
#inputStancesPath = "./fnc-1-original/train_stances_example.csv"
outputStancesPath = "./fnc-1-original/cleanDatasets/train_stances_clean.csv"
print(">>> Cleaning out Stances Data")
trainFile = pd.read_csv(inputStancesPath,header=0,delimiter=",", quoting=1)
# Limpiamos el texto
cleanTrainBodies = cleanData.cleanTextData(True,inputStancesPath, outputStancesPath, False)
# Creamos el modelo
(vectorizer, train_data_features) = createBOW.createBOWModel(cleanTrainBodies)


# Paso 2: Creamos el modelo de Train forest
# The random forest algorithm is included in scikit-learn. Random Forest uses many tree-based classifiers to
# make predictions, hence the "forest"
# Below, we set the number of trees to 100 as a reasonable default value. More trees may (or may not) perform
# better, but will certainly take longer to run

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response
# variable

# This may take a few minutes to run
print(">>> Fitting forest model")
forest = forest.fit(train_data_features, trainFile["Stance"])

# Paso 3: testeamos el modelo con el csv de test
testFilePath = "./fnc-1-original/test_stances_unlabeled.csv"
outputTestPath = "./fnc-1-original/cleanDatasets/test/test_stances_unlabeled_clean.csv"
testFile = pd.read_csv(testFilePath,header=0,delimiter=",", quoting=1)
print(">>> Testing RandomForest Model with dataset: ", trainFile.shape )

cleanHeadlines = cleanData.cleanTextData(True, testFilePath, outputTestPath)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(cleanHeadlines)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

#Copy the results to a pandas dataframse with an "id" column and
# a "Stance" column
output = pd.Dataframe(data= {"id": test["id"], "Stance": result})

# Use pandas to write the comma-separated output file
output.to_csv("Bag_of_words_model.csv", index=False, quoting=3)
