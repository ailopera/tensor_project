import sys
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from vectorAverage import executeVectorAverage
from BOWModel2 import generateBOWModel

# Script que realiza la validación de modelos, utilizando el método de k-fold con k=4 
validation = sys.argv[1]
# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
vectorAverage_iterations = [{"model": "300features_10minwords_10contextALL", "classifier": "MLP", "binaryModel": False}, \
        {"model": "300features_10minwords_10contextALL", "classifier": "MLP","binaryModel": False}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}, \
        {"model": "~/GoogleNews-vectors-negative300.bin", "classifier": "RF"}]

bow_iterations = [{ "min_df": 1.0, "max_df": 1}, \
        { "min_df": 0.1, "max_df": 1}, \
        { "min_df": 1.0, "max_df": 0.8}, \
        { "min_df": 0.1, "max_df": 0.8}, \
        { "min_df": 0.25, "max_df": 0.75}
]

trainPath = "train_partition.csv"
testPath = "test_partition.csv"

#cargamos el dataset de entrenamiento/validacion y el de test
trainDataPath = "./fnc-1-original/finalDatasets/train_partition.csv"
testDataPath = "./fnc-1-original/finalDatasets/test_partition.csv"
train_df = pd.read_csv(trainPath,header=0,delimiter=",", quoting=1)
test_df = pd.read_csv(testPath,header=0,delimiter=",", quoting=1)

# Get size of batch
k = 4
print(">> Performing K-Fold Validation with K ", k)
batch_size = len(train_df.iterrows()) / k 
print(">> K: ", k)
print(">> File samples: ", len(train_df.iterrows()))
print(">> Batch Size: ", batch_size)

# Get execution params based on implementation executed
if validation == "vectorAverage":
        iterations = vectorAverage_iterations
elif validation == "BOW":
        iterations = bow_iterations

print(">> Executing different model configurations over train data applying K-Fold Validation...")
for iteration in iterations:
    K_fold = KFold(k)
    # Execute k-fold validation k times
    #for k, (train_data,test_data) in enumerate(k_fold.split(train_df,target)):
    index = 1
    for train_indices, test_indices in K_fold.split(train_df):
        # 2. Fold the data between train and validation
        # X_train, X_test, y_train, y_test = train_test_split(train_df, train_df['Stances'], test_size=0.2)
        # train_data, test_data, train_targets, test_targets = train_test_split(train_df, train_df['Stances'], test_size=0.2)
        print(">>> Executin KFold iteration ", index)
        train_data, test_data = train_df[train_indices], train_df[test_indices]
        # train_targets, test_targets = train_df[train_indices]['Stance'], train_df[test_indices]['Stance']
        #3. Execute model with the configuration specified 
        if validation == "vectorAverage":
                executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"], train_data, test_data)
        elif validation == "BOW":
                generateBOWModel(trainData, test_data, iteration["min_df"], iteration["max_df"])
        index = index + 1


# Execute test with test data
# Execute the same iterations with final validation data
print("---------------------- TEST ------------------------------")
print(">> Executing different model configurations over test data")
# Creamos las dos particiones de datos
train_batch_size = math.ceil(len(test_df) * 0.8)
train_data = test_df[:train_batch_size]
test_data = test_df[train_batch_size:]
for iteration in iterations:
        if validation == "vectorAverage":
                executeVectorAverage(iteration["model"],iteration["classifier"], iteration["binaryModel"], train_data, test_data, True)
        elif validation == "BOW":
                generateBOWModel(trainData, testData, iteration["min_df"], iteration["max_df"], True)