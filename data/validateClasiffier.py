import sys, time, math, os, csv
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from vectorAverage import executeVectorAverage
from BOWModel2 import generateBOWModel

# Script que realiza la validación de modelos
# Uso: python validateClasiffier.py


# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
common_params = {"model": "300features_15minwords_10contextALL", "classifier": "MLP", "binaryModel": False, "smote": "all"}
iterations = [{ "activation_function": "relu", "config_tag": "prueba"}]

#cargamos el dataset de entrenamiento/validacion y el de test
# Primer particionado
#trainDataPath = "./fnc-1-original/finalDatasets/train_partition.csv"
#testDataPath = "./fnc-1-original/finalDatasets/test_partition.csv"
# Segundo particionado
trainDataPath = "./fnc-1-original/finalDatasets/train_partition_split.csv"
testDataPath = "./fnc-1-original/finalDatasets/test_partition_split.csv"
train_df = pd.read_csv(trainDataPath,header=0,delimiter=",", quoting=1)
test_df = pd.read_csv(testDataPath,header=0,delimiter=",", quoting=1)

# Get size of batch
start = time.time()

# Get execution params based on implementation executed


print(">> Executing different model configurations over train data applying simple validation...")

# Divide data between train and validation
# Run model with every configuration specified
train_proportion = math.ceil(train_df.shape[0] * 0.8)
train_data = train_df[:train_proportion]
validation_data = train_df[train_proportion:]
print(">>> LEN train data: ", len(train_data))
print(">>> LEN validation data: ", len(validation_data))

for classifier_config in iterations:
        print(">>> Executing Configuration: ", classifier_config)
        # Execute model with the configuration specified 
        executeVectorAverage(common_params["model"],common_params["classifier"], common_params["binaryModel"], train_data, validation_data,False, "all", classifier_config)
        print("------------------------------------------------------")

end = time.time()
trainValidationTime = end - start

start = time.time()
# Execute test with test data
# Execute the same iterations with final validation data
print("---------------------- TEST ------------------------------")
print(">> Executing different model configurations over test data")
print(">>> LEN train data: ", train_data.shape[0])
print(">>> LEN test data: ", test_df.shape[0])
#for iteration in iterations:
        #executeVectorAverage(common_params["model"],common_params["classifier"], common_params["binaryModel"], train_data, test_df, False, "all", classifier_config)
        
end = time.time()
testExecutionTime = end - start

print(">> TRAIN-VALIDATION EXECUTION TIME: ", trainValidationTime)
print(">> TEST EXECUTION TIME: ", testExecutionTime)

# Export data to a csv file
csvOutputDir = "./executionStats/"
date = time.strftime("%Y-%m-%d")
output_file =  csvOutputDir + "simpleValidation_execution_" + date + ".csv"
fieldNames = ["date", "execution", "trainValidationTime", "testTime"]

with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldNames)
        executionData = {
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "execution": "",
                "trainValidationTime": round(trainValidationTime,2),
                "testTime": round(testExecutionTime,2),
                }
                
        newFile = os.stat(output_file).st_size == 0
        if newFile:
                writer.writeheader()
        writer.writerow(executionData)

print(">> Stats exported to: ", output_file)
