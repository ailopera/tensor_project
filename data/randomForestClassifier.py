# Crea un clasificador random para testear los modelos de representacion de textos
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix 

def randomClassifier(trainDataFeatures, trainTargets, testDataFeatures, testTargets):
	# Creamos un modelo de random forest con los datos de entrenamiento, usando 100 árboles
	forest = RandomForestClassifier(n_estimators=100)

	print("> Fitting a random forest to labeled training data...")
	# print(">> TRAIN Lens: ArticleBody", len(trainDataVecsArticleBody), " Headline: ", len(trainDataVecsHeadline))
	# print(trainDataVecsArticleBody)
	# print("-----------------------------")
	# print(trainDataVecsHeadline)
	# print(">> TEST Lens: ArticleBody", len(testDataVecsArticleBody), " Headline: ", len(testDataVecsHeadline))
	# # trainDataFrame = pd.DataFrame.from_dict(trainDataVecs)
	# # trainDataFrame = pd.DataFrame({'Headline': trainDataVecsHeadline, 'ArticleBody': trainDataVecsArticleBody}, index=[0])
	# # features = trainDataFrame.columns[:2]
	# # features = ['Headline', 'ArticleBody']

	# forest = forest.fit(trainDataVecsArticleBody, trainData["Stance"])
	# forest = forest.fit([trainDataVecsHeadline, trainDataVecsArticleBody], trainData["Stance"])
	forest = forest.fit(trainDataFeatures, trainTargets)

	# Test & extract results
	print("> Predicting test dataset...")
	# testDataFrame = pd.DataFrame.from_dict(testDataVecs)
	# testDataFrame = pd.DataFrame.from_dict({'Headline': testDataVecsHeadline, 'ArticleBody': testDataVecsArticleBody}, index=[0])
	prediction = forest.predict(testDataFeatures)
	# prediction = forest.predict([testDataVecsHeadline, testDataVecsArticleBody])

	#  Evaluate the results
	# train_accuracy = accuracy_score(trainData['Stance'], forest.predict(trainDataVecs))
	# Creo dos modelos aparte como solucion temporal ya que con el random forest no es posible pasar los dos textos como features
	train_accuracy = accuracy_score(trainTargets, forest.predict(trainDataFeatures))
	test_accuracy = accuracy_score(testTargets, prediction)
	confussion_matrix = confusion_matrix(testTargets, prediction)
	
	print(">> Accuracy achieved with the train set (using only article bodies): ", train_accuracy)	
	print(">> Accuracy achieved with the test set: ", test_accuracy)
	print(">> Confussion matrix: ", confusion_matrix)
	
	# # Write the test results
	# outputFile = "Word2Vec_AverageVectors.csv"
	# print(">> Generating output file : ", outputFile)
	# output = pd.DataFrame(data={"id": test["id"], "stance": prediction})
	# output.to_csv(outputFile, index=False, quoting=3)
