import pandas as pd
from sklearn.model_selection import KFold

import sys

from vectorAverage import executeVectorAverage
# Script que realiza la validación de modelos, utilizando el método de k-fold con k=4 

# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
iterations = [{"model": "300features_10minwords_10contextALL", "classifier": "MLP", "binaryModel": False}, \
        {"model": "300features_10minwords_10contextALL", "classifier": "MLP","binaryModel": False}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}]


trainPath = "train_partition.csv"
testPath = "test_partition.csv"

#cargamos el dataset de entrenamiento/validacion
train_df = pd.read_csv(trainPath,header=0,delimiter=",", quoting=1)

# X = train_df[['Headline']]#Datos de entrenamiento
target = train_df["Stances"] # Columna de Stances
test_df = pd.read_csv(testPath,header=0,delimiter=",", quoting=1)

data = sys.argv[1]
k = 4
for iteration in iterations:
    # 2. Get size of batch
    batch_size = len(data.iterrows()) / k 
    print(">> K: ", k)
    print(">> File samples: ", len(data.iterrows()))
    print(">> Batch Size: ", batch_size)
    
    K_fold = KFold(k)
    # Execute k-fold validation k times
    for k, (train_data,test_data) in enumerate(k_fold.split(train_df,target)):
        # 2. Fold the data between train and validation
        X_train, X_test, y_train, y_test = train_test_split(df, stances, test_size=0.2)
        #3. Execute vectorAverage model with the configuration specified 
        executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"],train_data, test_data, True)


# Execute test with test data
# Execute the same iterations with final validation data

# # TODO: Cargamos el dataset de test
# for iteration in iterations:
#     executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"],train_data, test_data)