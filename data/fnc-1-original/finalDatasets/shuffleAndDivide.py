import random
import csv
import pandas as pd

# Pequeño script que se encarga de realizar las particiones de datos que se van a utilizar en este trabajo

inputPath = "total_data_aggregated.csv"
# header = "Headline,ArticleBody,Stance,BodyIDS"
outputTrainPath = "train_partition.csv"
outputTestPath = "test_partition.csv"

print(">> Fichero de entrada:", inputPath)
# inputFile = pd.read_csv(inputPath,header=0,delimiter=",", quoting=1)
df = pd.read_csv(inputPath,header=0,delimiter=",", quoting=1)

with open(outputTrainPath, 'w') as trainFile, open(outputTestPath, 'w') as testFile:
    print(">> Fichero de train generado:", outputTrainPath)
    print(">> Fichero de test generado:", outputTestPath)
    
    fieldnames = ["Headline","Body ID","Stance"]
    trainWriter = csv.DictWriter(trainFile, fieldnames=fieldnames)
    testWriter = csv.DictWriter(testFile, fieldnames=fieldnames)
    
    # Escribimos los headers de los dos datasets
    testWriter.writeheader()
    trainWriter.writeheader()

    # data = inputFile.iterrows()
    # Agitamos los datos para crear una partición de datos aleatoria
    print(">> Aleatorizando las muestras...")
    # random.shuffle(data)
    df.sample(frac=1)

    # Creamos las dos particiones de datos
    train_data = df[:80]
    test_data = df[80:]
    print(">> Muestras de entrenamiento y validación: ", len(train_data))
    print(">> Muestras de testeo del modelo: ", len(test_data))

    # Escribimos las particiones en los ficheros correspondientes
    print(">> Escribiendo los datos en los ficheros de salida correspondientes...")
    trainWriter.writerows(train_data)
    testWriter.writerows(test_data)


    

    