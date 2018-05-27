import random
import csv
import pandas as pd
import math

# Pequeño script que se encarga de realizar las particiones de datos que se van a utilizar en este trabajo

#Particionado 3
inputPath = "competition_data_aggregated_sorted.csv"
# header = "Headline,ArticleBody,Stance,BodyIDS"
#outputTrainPath = "train_partition_3.csv"
outputTestPath = "test_partition_3.csv"

print(">> Fichero de entrada:", inputPath)
# inputFile = pd.read_csv(inputPath,header=0,delimiter=",", quoting=1)
df = pd.read_csv(inputPath,header=0,delimiter=";", quoting=1)

with open(outputTestPath, 'w') as testFile:
    print(">> Fichero de test generado:", outputTestPath)
    
    fieldnames = ["Headline","ArticleBody","Stance","BodyIDS"]

    testWriter = csv.DictWriter(testFile, fieldnames=fieldnames)
    
    # Escribimos los headers de los dos datasets
    testWriter.writeheader()

    # data = inputFile.iterrows()
    
    # Creamos las dos particiones de datos
    #Particionado 2
    #train_batch_size = math.ceil(len(df) * 0.8)
    # Particionado 3 (Como solo utilizaremos la de test, cambiamos la division entre train/test)
    test_batch_size = math.ceil(len(df) * 0.5)
    test_data = df[:test_batch_size]
    print(">> Muestras de testeo del modelo: ", len(test_data))

    # Agitamos los datos para desordenar el orden de las muestras
    print(">> Aleatorizando las muestras...")
    df.sample(frac=1)

    print(">> Escribiendo los datos en el fichero de test...")
    for index,line in test_data.iterrows():
        row = { "Headline": line["Headline"],
        "ArticleBody": line["ArticleBody"],
        "Stance": line["Stance"],
        "BodyIDS": line["BodyIDS"]}

        testWriter.writerow(row)
        