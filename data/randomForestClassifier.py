# Crea un clasificador random para testear los modelos de representacion de textos
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np

def convert_to_int_classes(targetList):
    map = {
        "agree": 0,
        "disagree": 1,
        "discuss": 2,
        "unrelated": 3
    }
    int_classes = []
    for elem in targetList:
        int_classes.append(map[elem])
        
    return np.array(int_classes)


def randomClassifier(trainDataFeatures, trainTargets, testDataFeatures, testTargets):
	# Convertimos a enteros las clases
    train_labels = convert_to_int_classes(trainTargets)
    test_labels = convert_to_int_classes(testTargets)
	
	# Creamos un modelo de random forest con los datos de entrenamiento, usando 100 árboles
    forest = RandomForestClassifier(n_estimators=100)
    print("> Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataFeatures, train_labels)

	# Test & extract results
    print("> Predicting test dataset...")
    prediction = forest.predict(testDataFeatures)

	# Metricas de entrenamiento
    train_accuracy = accuracy_score(train_labels, forest.predict(trainDataFeatures))
    precision_train = precision_score(train_labels, forest.predict(trainDataFeatures), average='weighted', labels=[0,1,2,3])
    recall_train = recall_score(train_labels, forest.predict(trainDataFeatures), average='weighted', labels=[0,1,2,3])
    
    # Metricas de test
    test_accuracy = accuracy_score(test_labels, prediction)
    precision_test = precision_score(test_labels, prediction, average='weighted', labels=[0,1,2,3])
    recall_test = recall_score(test_labels, prediction, average='weighted', labels=[0,1,2,3])

    confussion_matrix = confusion_matrix(test_labels, prediction,labels=[0,1,2,3])

    print(">> Accuracy achieved with the train set: ", train_accuracy)	
    print(">> Accuracy achieved with the test set: ", test_accuracy)
    print(">> Precision test: ", precision_train)
    print(">> Precision test: ", precision_test)
    print(">> Recall train: ", recall_train)
    print(">> Recall test: ", recall_test)

    print(">> Confussion matrix: ")
    print(confussion_matrix)

    y_score = forest.predict_proba(test_labels)

    metrics = {
	 	"train_accuracy": round(train_accuracy,2),
		"test_accuracy": round(test_accuracy,2),
		"precision_train": round(precision_train,2),
        "precision_test": round(precision_test,2),
        "recall_train": round(recall_train,2),
        "recall_test": round(recall_test,2),
        "confusion_matrix": confussion_matrix,
        "y_true": test_labels,
        "y_score": y_score
		}
	
    return metrics
	
	# # Write the test results
	# outputFile = "Word2Vec_AverageVectors.csv"
	# print(">> Generating output file : ", outputFile)
	# output = pd.DataFrame(data={"id": test["id"], "stance": prediction})
	# output.to_csv(outputFile, index=False, quoting=3)
