import sys
import pandas as pd
from sklearn.model_selection import KFold

from vectorAverage import executeVectorAverage
from BOWModel2 import generateBOWModel

# Script que realiza la validación de modelos, utilizando el método de k-fold con k=4 
validation = sys.argv[1]
# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
vectorAverage_iterations = [{"model": "300features_10minwords_10contextALL", "classifier": "MLP", "binaryModel": False}, \
        {"model": "300features_10minwords_10contextALL", "classifier": "MLP","binaryModel": False}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}]

bow_iterations = [
        { "min_df": 1.0, "max_df": 1}, \ 
        { "min_df": 0.1, "max_df": 1}, \
        { "min_df": 1.0, "max_df": 0.8}, \
        { "min_df": 0.1, "max_df": 0.8}, \
        { "min_df": 0.25, "max_df": 0.75}
]

trainPath = "train_partition.csv"
testPath = "test_partition.csv"

#cargamos el dataset de entrenamiento/validacion
train_df = pd.read_csv(trainPath,header=0,delimiter=",", quoting=1)

# X = train_df[['Headline']]#Datos de entrenamiento
target = train_df["Stances"] # Columna de Stances
test_df = pd.read_csv(testPath,header=0,delimiter=",", quoting=1)

data = sys.argv[1]
# 2. Get size of batch
k = 4
print(">> Performing K-Fold Validation with K ", k)
batch_size = len(data.iterrows()) / k 
print(">> K: ", k)
print(">> File samples: ", len(data.iterrows()))
print(">> Batch Size: ", batch_size)

# Get execution params based on implementation executed
if validation == "vectorAverage":
        iterations = vectorAverage_iterations
elif validation == "BOW":
        iterations = bow_iterations

for iteration in iterations:
    K_fold = KFold(k)
    # Execute k-fold validation k times
    for k, (train_data,test_data) in enumerate(k_fold.split(train_df,target)):
        # 2. Fold the data between train and validation
        X_train, X_test, y_train, y_test = train_test_split(df, stances, test_size=0.2)
        #3. Execute model with the configuration specified 
        if validation == "vectorAverage":
                executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"], train_data, test_data, True)
        elif validation == "BOW":
                generateBOWModel(trainData, testData, False)

# Execute test with test data
# Execute the same iterations with final validation data

# # TODO: Cargamos el dataset de test
# for iteration in iterations:
#     executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"],train_data, test_data)