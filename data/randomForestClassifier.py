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

	#  Evaluate the results
    train_accuracy = accuracy_score(train_labels, forest.predict(trainDataFeatures))
    test_accuracy = accuracy_score(test_labels, prediction)
    confussion_matrix = confusion_matrix(test_labels, prediction,labels=[0,1,2,3])
    average_precision = precision_score(test_labels, prediction, average='weighted', labels=[0,1,2,3])
    recall = recall_score(test_labels, prediction, average='weighted', labels=[0,1,2,3])

    print(">> Accuracy achieved with the train set: ", train_accuracy)	
    print(">> Accuracy achieved with the test set: ", test_accuracy)
    print(">> Confussion matrix: ")
    print(confussion_matrix)
    print(">> Average precision (micro): ", average_precision)
    print(">> Recall: ", recall)

    y_score = forest.predict_proba(test_labels)

    metrics = {
	 	"train_accuracy": round(train_accuracy,2),
		"test_accuracy": round(test_accuracy,2),
		"confusion_matrix": confussion_matrix,
		"precision_test": round(average_precision,2),
		"precision_train": 0,
        "recall_test": round(recall,2),
        "precision_test": 0,
        "y_true": test_labels,
        "y_score": y_score
		}
	
    return metrics
	
	# # Write the test results
	# outputFile = "Word2Vec_AverageVectors.csv"
	# print(">> Generating output file : ", outputFile)
	# output = pd.DataFrame(data={"id": test["id"], "stance": prediction})
	# output.to_csv(outputFile, index=False, quoting=3)
