# Pequeno script que hace un agregado de los datos de entrada, 
# para tratarlos más adelante

import pandas as pd
import sys
import cleanData
import csv

if __name__ == "__main__":
    
    baseFilename = sys.argv[1]
    if baseFilename == "competition":
        filePath = "competition_test_$file"
    elif baseFilename == "train":
        filePath = "train_$file"
    elif baseFilename == "test":
        filePath = "test_$file"
    else:
        print("> Wrong name supplied. Must be competition, train or test")
        exit()

    # Primera parte del estudio: experimentacion con modelos de representacion y MLP
    # inputPath = "./fnc-1-original/"
    # outputPath = inputPath + "aggregatedDatasets/"

    # Segunda parte del estudio: experimentacion con flujos de texto y RNN
    inputPath = "./fnc-1-original/"
    outputPath = inputPath + "aggregatedDatasets/RNN/"

    # Cargamos ficheros de stances y bodies 
    stancesPrefixPath = filePath.replace('$file','stances')
    bodiesPrefixPath = filePath.replace('$file','bodies')
    
    inputBodyFile = inputPath + bodiesPrefixPath + ".csv"
    inputStanceFile = inputPath + stancesPrefixPath + ".csv"

    outputBodyFile = outputPath + bodiesPrefixPath + "_clean.csv"
    outputStanceFile = outputPath + stancesPrefixPath + "_clean.csv"
    aggregatedFile = outputPath + baseFilename + "_data_aggregated.csv"
    
    #Limpiamos fichero de bodies y stances
    # 1. Generacion de datasets para la experimentacion de MLP
    # print(">> Limpiando fichero de bodies ", inputBodyFile, " y generando fichero ", outputBodyFile)
    # cleanData.cleanTextData(False,inputBodyFile,outputBodyFile,False)
    # print(">> Limpiando fichero de bodies ", inputStanceFile, " y generando fichero ", outputStanceFile)
    # cleanData.cleanTextData(True,inputStanceFile,outputStanceFile,False)


    # 2. Generacion de datasets para la experimentacion de RNN
    print(">> Limpiando fichero de bodies ", inputBodyFile, " y generando fichero ", outputBodyFile)
    cleanData.cleanTextData(False,inputBodyFile,outputBodyFile, printLogs=False, maintainDots = True)
    print(">> Limpiando fichero de bodies ", inputStanceFile, " y generando fichero ", outputStanceFile)
    cleanData.cleanTextData(True,inputStanceFile,outputStanceFile, printLogs=False, maintainDots = True)


    # Abrimos el fichero de stances y lo vamos recorriendo para hacer el agregado de los datos
    cleanedStanceData = pd.read_csv(outputStanceFile, header=0,delimiter=",", quoting=1)
    cleanedBodyData = pd.read_csv(outputBodyFile, header=0,delimiter=",", quoting=1, index_col='Body ID')

    
    # bodyFileDict = loadBodyDict(cleanedBodyData)
    print(">> Escribiendo resultados en ", aggregatedFile)
    with open(aggregatedFile, 'w') as aggregatedData:
        fieldnames = ["Headline","ArticleBody", "Stance", "BodyIDS"]
        
        writer = csv.DictWriter(aggregatedData, fieldnames=fieldnames)
        writer.writeheader()
        i = 0
        for index, line in cleanedStanceData.iterrows():
            # Buscamos el cuerpo asociado
            bodyId = line["Body ID"]
            # associatedBody = cleanedBodyData.loc[cleanedBodyData['Body ID'] == bodyId, 'Body ID']
            associatedBody = cleanedBodyData.loc[bodyId]
            # associatedBody = cleanedBodyData.loc[cleanedBodyData['Body ID'] == bodyId]
            #associatedBodyId = cleanedBodyData.loc[cleanedBodyData['Body ID'] == bodyId, 'Body ID']
            #print(">>> type(associatedBody): ", type(associatedBody))
            # if len(associatedBody) == 0:
            #     print(">> ERROR: He encontrado ", len(associatedBody), " elementos")
            #     print(">> Associated Body: ", associatedBody)
            # else: 
            
            aggregatedLine = {
                "Headline": line["Headline"],
                "ArticleBody": associatedBody.get('articleBody'),
                "Stance": line["Stance"],
                "BodyIDS": bodyId
                # "BodyIDB": associatedBody.get('Body ID'),
            }
            # Escribimos la línea en el fichero
            writer.writerow(aggregatedLine)
            if i%1000 == 0:
                print(">> Counter: ", i)
                print(">> Associated Body: ", associatedBody)
                print(">> ROW: ", aggregatedLine)
                print("-------------------------------------")
            i +=1
