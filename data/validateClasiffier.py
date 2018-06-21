import sys, time, math, os, csv
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from vectorAverage import executeVectorAverage
from BOWModel2 import generateBOWModel
from embeddingVector import executeVectorFeaturing
# Script que realiza la validación de modelos
# Uso: python validateClasiffier.py


# Definimos las distintas configuraciones con las que evaluaremos el modelo. Cada configuración se evalúa k veces
common_params = {"representation":"vectorAverage","model": "300features_15minwords_10contextALL", "classifier": "MLP", "binaryModel": False, "smote": "all"}
#common_params = {"representation": "vectorAverage","model": "~/GoogleNews-vectors-negative300.bin", "classifier": "MLP", "binaryModel": True, "smote": "all"}

#common_params = { "representation": "BOW", "classifier": "MLP", "min_df": 1, "max_df": 1.0, "smote": "all"}

# configuraciones antiguas
# iterations = [
        # { "activation_function": "relu", "config_tag": "original", "hidden_neurons": [300, 100]}, #Configuracion original
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [400, 200]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [500, 300]},
    
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 25},    
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 30},
        #{ "activation_function": "relu", "config_tag": "epochs", "hidden_neurons": [500, 300], "epochs": 35},
        
        #{ "activation_function": "relu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        #{ "activation_function": "leaky_relu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        #{ "activation_function": "elu", "config_tag": "activation", "hidden_neurons": [500, 300]},
        
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [550, 350]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [400, 400]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [300, 300]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [200, 100]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [200, 50]},
        #{ "activation_function": "relu", "config_tag": "neurons", "hidden_neurons": [50, 25]},

        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 225, 150]}, # 3 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [500, 275, 200]}, # 3 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 300, 225, 150]}, # 4 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375, 225, 150, 75]}, # 4 capas
        #{ "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 375, 300, 225, 150]}, # 5 capas
        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375,300, 225, 150, 75]},  # 5 capas

        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [450, 225, 150], "learning_rate": 0.1}, # 3 capas
        # { "activation_function": "relu", "config_tag": "layers", "hidden_neurons": [375, 225, 150, 75], "learning_rate": 0.1}, # 4 capas
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [375,300, 225, 150, 75], "learning_rate": 0.1} # 5 capas

        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.25},
        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.50},
        #{ "activation_function": "relu", "config_tag": "dropout", "hidden_neurons": [300, 100], "dropout_rate": 0.75}

        # { "activation_function": "relu", "config_tag": "architecture_shallow", "hidden_neurons": [500, 250],"learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [100, 100, 100, 100], "learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [150, 125, 100, 75], "learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [125, 100, 100, 75, 50],"learning_decrease": True, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "architecture_deep", "hidden_neurons": [100, 75, 50, 50, 25], "learning_decrease": True, "learning_rate": 0.1} 
        
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [300, 100],"dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [500, 250],"dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [250, 100, 50], "dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [150, 125, 100, 75], "dropout_rate": 0.50},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers", "hidden_neurons": [125, 100, 100, 75, 50], "dropout_rate": 0.50}, 

        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [300, 100],"dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [500, 250],"dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [250, 100, 50], "dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [150, 125, 100, 75], "dropout_rate": 0.75},
        # { "activation_function": "relu", "config_tag": "dropout_hidden_layers_25", "hidden_neurons": [125, 100, 100, 75, 50], "dropout_rate": 0.75},
        
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [500, 250], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [100, 100, 100, 100], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [150, 125, 100, 75], "learning_decrease": 0.9, "learning_rate": 0.1},
        # { "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [125, 100, 100, 75, 50],"learning_decrease": 0.9, "learning_rate": 0.1}
        #{ "activation_function": "relu", "config_tag": "learning_rate", "hidden_neurons": [100, 75, 50, 50, 25], "learning_decrease": 0.9, "learning_rate": 0.1}          # Ejecuciones aumentando el tamano del batch
        #{ "activation_function": "relu", "config_tag": "ampliando_batch", "hidden_neurons": [300, 100], 'batch_size': 150}  
# ]

# Experimentación final
base_arquitecture = [300, 100]
#base_arquitecture = [300, 100, 100]
#base_arquitecture = [250,150,100,75]
#base_arquitecture = [200,100,100,75]
#base_arquitecture = [250,150,100]
iterations = [
        # Ejecuciones base
        #{ "activation_function": "relu", "config_tag": "base_arquitecture", "hidden_neurons": base_arquitecture}, #Configuracion original
        #{ "activation_function": "relu", "config_tag": "reduccion_neuronas", "hidden_neurons": [200, 50]}, # Reduciendo el numero de neuronas
        
        # Early Stopping sobre la arquitectura base

        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_2", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.01, "early_stopping_patience": 2}, 
        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_1.5", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.01, "early_stopping_patience": 1.5},
        #{ "activation_function": "relu", "config_tag": "base_arquitecture_early_stopping_3", "hidden_neurons": base_arquitecture, "early_stopping": True, "learning_rate": 0.01, "early_stopping_patience": 3},

        # Ejecuciones aplicando regularización Dropout
        #{ "activation_function": "relu", "config_tag": "dropout_25", "hidden_neurons": base_arquitecture, "dropout_rate": 0.25, "epochs": 20, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "dropout_35", "hidden_neurons": base_arquitecture, "dropout_rate": 0.35, "epochs": 20, "learning_rate": 0.05},
        #{ "activation_function": "relu", "config_tag": "dropout_50", "hidden_neurons": base_arquitecture, "dropout_rate": 0.50, "epochs": 20, "learning_rate": 0.05}, 
        #{ "activation_function": "relu", "config_tag": "dropout_75", "hidden_neurons": base_arquitecture, "dropout_rate": 0.75, "epochs": 20, "learning_rate": 0.05}, 
        # Ejecuciones aplicando regularización L2
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.001", "hidden_neurons": base_arquitecture, "l2_scale": 0.001, "learning_rate": 0.05},
        { "activation_function": "relu", "config_tag": "l2_scale_0.002", "hidden_neurons": base_arquitecture, "l2_scale": 0.002, "learning_rate": 0.05}
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.005", "hidden_neurons": base_arquitecture, "l2_scale": 0.005, "learning_rate": 0.05}
        #{ "activation_function": "relu", "config_tag": "l2_scale_0.007", "hidden_neurons": base_arquitecture, "l2_scale": 0.007, "learning_rate": 0.01}
        
        
        
        # Ejecuciones aumentando el numero de capas
        #{ "activation_function": "relu", "config_tag": "ampliacion_capa", "hidden_neurons": [300, 100, 100]},
        #{ "activation_function": "relu", "config_tag": "ampliacion_capa", "hidden_neurons": [300, 100, 100]},
        #{ "activation_function": "relu", "config_tag": "ampliacion_capa_2", "hidden_neurons": [300, 100, 100, 50]},
        
        #Pruebas combinadas
        # { "activation_function": "relu", "config_tag": "ampliacion_capa_dropout", "hidden_neurons": [300, 100, 100], "dropout_rate": 0.25, "epochs": 20, "learning_rate": 0.05},
        # { "activation_function": "relu", "config_tag": "ampliacion_capa_l2_scale", "hidden_neurons": [300, 100, 100], "epochs": 20, "l2_scale": 0.001, "learning_rate": 0.01},
        # { "activation_function": "relu", "config_tag": "ampliacion_capa_2_dropout", "hidden_neurons": [300, 100, 100, 50], "dropout_rate": 0.25, "epochs": 20, "learning_rate": 0.05},
        # { "activation_function": "relu", "config_tag": "ampliacion_capa_2_l2_scale", "hidden_neurons": [300, 100, 100, 50], "epochs": 20, "l2_scale": 0.001, "learning_rate": 0.01}
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
           executeVectorAverage(common_params["model"],common_params["classifier"], common_params["binaryModel"], train_data, validation_data,False, common_params["smote"], classifier_config)
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
