import sys, time, math, os, csv
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from vectorAverage import executeVectorAverage
from BOWModel2 import generateBOWModel
from embeddingVector import executeVectorFeaturing
# Script que realiza la validación de modelos
# Uso: python validateClasiffier.py


# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
# common_params = {"representation":"vectorAverage","model": "300features_15minwords_10contextALL", "classifier": "MLP", "binaryModel": False, "smote": "all"}
common_params = {"representation": "vectorAverage","model": "~/GoogleNews-vectors-negative300.bin", "classifier": "MLP", "binaryModel": True, "smote": "all"}

#common_params = { "representation": "BOW", "classifier": "MLP", "min_df": 1, "max_df": 1.0, "smote": "all"}

# Experimentación final
base_arquitecture = [300, 100]
#base_arquitecture = [300, 100, 100]
#base_arquitecture = [250,150,100,75]
#base_arquitecture = [200,100,100,75]
#base_arquitecture = [250,150,100]
iterations = [
        # Ejecuciones base
        #{ "activation_function": "relu", "config_tag": "base_arquitecture", "hidden_neurons": base_arquitecture},
        
        # Gradient Descent +  Learning rate dinamico
        { "activation_function": "relu", "config_tag": "dynamic_learning_rate_0.01_0.95", "hidden_neurons": base_arquitecture, "learning_rate": 0.01, "learning_decrease_base":0.95},
        { "activation_function": "relu", "config_tag": "dynamic_learning_rate_0.01_0.90", "hidden_neurons": base_arquitecture, "learning_rate": 0.01, "learning_decrease_base":0.90},
        { "activation_function": "relu", "config_tag": "dynamic_learning_rate_0.05_0.95", "hidden_neurons": base_arquitecture, "learning_rate": 0.05, "learning_decrease_base":0.95},
        { "activation_function": "relu", "config_tag": "dynamic_learning_rate_0.05_0.90", "hidden_neurons": base_arquitecture, "learning_rate": 0.05, "learning_decrease_base":0.90},
        
        # Aplicando optimizador ADAM
        { "activation_function": "relu", "config_tag": "base_arquitecture_default_ADAM", "hidden_neurons": base_arquitecture, "learning_rate": 0.001,"optimizer_function": "ADAM"},
        { "activation_function": "relu", "config_tag": "base_arquitecture_default_ADAM", "hidden_neurons": base_arquitecture, "learning_rate": 0.01,"optimizer_function": "ADAM"},
        { "activation_function": "relu", "config_tag": "base_arquitecture_default_ADAM", "hidden_neurons": base_arquitecture, "learning_rate": 0.015,"optimizer_function": "ADAM"},
        
        # Aplicando optimizador Momentum
        

        # Early Stopping sobre la arquitectura base
        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_2", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.001, "early_stopping_patience": 2}, 
        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_1.5", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.001, "early_stopping_patience": 1.5},
        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_3", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.001, "early_stopping_patience": 3},

        # Ejecuciones aplicando regularización Dropout
        #{ "activation_function": "relu", "config_tag": "dropout_25", "hidden_neurons": base_arquitecture, "dropout_rate": 0.25, "epochs": 20, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "dropout_35", "hidden_neurons": base_arquitecture, "dropout_rate": 0.35, "epochs": 20, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "dropout_50", "hidden_neurons": base_arquitecture, "dropout_rate": 0.50, "epochs": 20, "learning_rate": 0.05}, 
        
        # Ejecuciones aplicando regularización L2
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.001", "hidden_neurons": base_arquitecture, "l2_scale": 0.001, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.002", "hidden_neurons": base_arquitecture, "l2_scale": 0.002, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.005", "hidden_neurons": base_arquitecture, "l2_scale": 0.005, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.007", "hidden_neurons": base_arquitecture, "l2_scale": 0.007, "learning_rate": 0.01}
        
]

# Configuraciones del clasificador recurrente
#iterations = [ 
#   {"recurrrent": True, "architecture": "simple", "config_tag": "RNN_simple"},
#  {"recurrrent": True, "architecture": "multi", "config_tag": "RNN multicapa"}
#]

#cargamos el dataset de entrenamiento/validacion y el de test
# Primer particionado
#trainDataPath = "./fnc-1-original/finalDatasets/train_partition.csv"
#testDataPath = "./fnc-1-original/finalDatasets/test_partition.csv"
# Segundo particionado
trainDataPath = "./fnc-1-original/finalDatasets/train_partition_split.csv"
testDataPath = "./fnc-1-original/finalDatasets/test_partition_split.csv"

# Tercer particionado (Solo cambia la particion de test)
# trainDataPath = "./fnc-1-original/finalDatasets/train_partition_split.csv"
# testDataPath = "./fnc-1-original/finalDatasets/test_partition_3.csv"
#testDataPath = "./fnc-1-original/finalDatasets/competition_data_aggregated.csv"

# Particionado para la experimentacion con RNN (Dataset con simbolos de puntuacion)
#trainDataPath = "./fnc-1-original/finalDatasets/RNN/train_partition_split.csv"
#testDataPath = "./fnc-1-original/finalDatasets/RNN/test_partition_split.csv"


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
        if common_params['representation'] == 'vectorAverage':
           executeVectorAverage(common_params["model"],common_params["classifier"], common_params["binaryModel"], train_data, validation_data,True, common_params["smote"], classifier_config)
            #executeVectorFeaturing(common_params["model"],common_params["classifier"], common_params["binaryModel"], train_data, validation_data,False, common_params["smote"], classifier_config)
        # elif common_params['representation'] == 'BOW':
        #   generateBOWModel(common_params["classifier"], train_data, validation_data, common_params["min_df"], common_params["max_df"],False, common_params["smote"])
        print("------------------------------------------------------")

end = time.time()
trainValidationTime = end - start

start = time.time()
# Execute test with test data
# Execute the same iterations with final validation data
# print("---------------------- TEST ------------------------------")
# print(">> Executing different model configurations over test data")
# print(">>> LEN train data: ", train_data.shape[0])
# print(">>> LEN test data: ", test_df.shape[0])
# for iteration in iterations:
        
#         print(">>> Executing Configuration: ", iteration)
#         # Execute model with the configuration specified 
#         if common_params['representation'] == 'vectorAverage':
#         #   executeVectorAverage(common_params["model"], common_params["classifier"], common_params["binaryModel"], train_data, test_df,False, common_params["smote"], iteration)
#             executeVectorFeaturing(common_params["model"], common_params["classifier"], common_params["binaryModel"], train_data, test_df,False, common_params["smote"], iteration)
#         # elif common_params['representation'] == 'BOW':
#         #   generateBOWModel(common_params["classifier"], train_data, test_df, common_params["min_df"], common_params["max_df"],False, common_params["smote"])
#         print("------------------------------------------------------")

# end = time.time()
# testExecutionTime = end - start

print(">> TRAIN-VALIDATION EXECUTION TIME: ", trainValidationTime)
#print(">> TEST EXECUTION TIME: ", testExecutionTime)

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
