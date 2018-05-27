import random
import csv
import pandas as pd
import math

# Pequeño script que se encarga de realizar las particiones de datos que se van a utilizar en este trabajo

#Particionado 3
inputPath = "competition_data_aggregated_sorted.csv"
# header = "Headline,ArticleBody,Stance,BodyIDS"
outputTrainPath = "train_partition_3.csv"
outputTestPath = "test_partition_3.csv"

print(">> Fichero de entrada:", inputPath)
# inputFile = pd.read_csv(inputPath,header=0,delimiter=",", quoting=1)
df = pd.read_csv(inputPath,header=0,delimiter=",", quoting=1)

with open(outputTrainPath, 'w') as trainFile, open(outputTestPath, 'w') as testFile:
    print(">> Fichero de train generado:", outputTrainPath)
    print(">> Fichero de test generado:", outputTestPath)
    
    fieldnames = ["Headline","ArticleBody","Stance","BodyIDS"]

    trainWriter = csv.DictWriter(trainFile, fieldnames=fieldnames)
    testWriter = csv.DictWriter(testFile, fieldnames=fieldnames)
    
    # Escribimos los headers de los dos datasets
    testWriter.writeheader()
    trainWriter.writeheader()

    # data = inputFile.iterrows()
    
    # Creamos las dos particiones de datos
    #Particionado 2
    #train_batch_size = math.ceil(len(df) * 0.8)
    # Particionado 3 (Como solo utilizaremos la de test, cambiamos la division entre train/test)
    train_batch_size = math.ceil(len(df) * 0.5)
    train_data = df[:train_batch_size]
    test_data = df[train_batch_size:]
    print(">> Muestras de entrenamiento y validación: ", len(train_data))
    print(">> Muestras de testeo del modelo: ", len(test_data))

    # Agitamos los datos para desordenar el orden de las muestras
    print(">> Aleatorizando las muestras...")
    # random.shuffle(data)
    df.sample(frac=1)

    
    # Escribimos las particiones en los ficheros correspondientes
    print(">> Escribiendo los datos en el fichero de entrenamiento/validacion...")
    for index,line in train_data.iterrows():
        row = { "Headline": line["Headline"],
        "ArticleBody": line["ArticleBody"],
        "Stance": line["Stance"],
        "BodyIDS": line["BodyIDS"]}

        trainWriter.writerow(row)
    
    print(">> Escribiendo los datos en el fichero de test...")
    for index,line in test_data.iterrows():
        row = { "Headline": line["Headline"],
        "ArticleBody": line["ArticleBody"],
        "Stance": line["Stance"],
        "BodyIDS": line["BodyIDS"]}

        testWriter.writerow(row)

    

    