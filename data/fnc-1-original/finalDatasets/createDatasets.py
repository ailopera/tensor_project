import random
import csv
import pandas as pd
import math

# Pequeño script que se encarga de realizar las particiones de datos que se van a utilizar en este trabajo
# NOTA: Este script comprueba que se utilicen cuerpos de noticia unicos en cada particion

inputPath = "total_data_aggregated.csv"
# header = "Headline,ArticleBody,Stance,BodyIDS"
# Primera parte de la experimentacion: BOW,vectorAverage y MLP
# outputTrainPath = "train_partition_split.csv"
# outputTestPath = "test_partition_split.csv"

# Segunda parte de la experimentacion: flujos de texto y RNN
outputTrainPath = "RNN/train_partition_split.csv"
outputTestPath = "RNN/test_partition_split.csv"

bodyIds = list(range(2532))
#print(bodyIds)
randomizedIds = random.shuffle(bodyIds)

train_batch_size = math.ceil(len(bodyIds) * 0.8)
train_ids = bodyIds[:train_batch_size]
test_ids = bodyIds[train_batch_size:]
print(">> Noticias distintas para el entrenamiento y validación: ", len(train_ids))
print(">> Noticias distintas para el testeo del modelo: ", len(test_ids))

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

    # Escribimos las particiones en los ficheros correspondientes
    for index,line in df.iterrows():
        row = { "Headline": line["Headline"],
        "ArticleBody": line["ArticleBody"],
        "Stance": line["Stance"],
        "BodyIDS": line["BodyIDS"]}

        bodyIds = int(line["BodyIDS"]) if not line[3] == "BodyIDS" else -1
        if bodyIds in train_ids:
            trainWriter.writerow(row)
        elif bodyIds in test_ids: 
            testWriter.writerow(row)
        else:
            print("Id no existente en la lista: ", bodyIds)