# Pequeno script que hace un agregado de los datos de entrada, 
# para tratarlos más adelante

import pandas as pd
import sys
import cleanData

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

    inputPath = "./fnc-1-original/"
    outputPath = inputPath + "/aggregatedDatasets/"

    # Cargamos ficheros de stances y bodies 
    stancesPrefixPath = filePath.replace('$file','stances')
    bodiesPrefixPath = filePath.replace('$file','bodies')

    #Limpiamos fichero de bodies y stances
    inputBodyFile = inputPath + bodiesPrefixPath + ".csv"
    inputStanceFile = inputPath + stancesPrefixPath + ".csv"
    
    outputBodyFile = outputPath + bodiesPrefixPath + "_clean.csv"
    outputStanceFile = outputPath + stancesPrefixPath + "_clean.csv"
    
    print(">> Limpiando fichero de bodies ", inputBodyFile, " y generando fichero ", outputBodyFile)
    cleanData.cleanTextData(False,inputBodyFile,outputBodyFile,True)
    print(">> Limpiando fichero de bodies ", inputStanceFile, " y generando fichero ", outputStanceFile)
    cleanData.cleanTextData(True,inputStanceFile,outputStanceFile,True)

    # Abrimos el fichero de stances y lo vamos recorriendo para hacer el agregado de los datos
    cleanedStanceData = pd.read_csv(outputStanceFile, header=0,delimiter=",", quoting=1)
    cleanedBodyData = pd.read_csv(outputBodyFile, header=0,delimiter=",", quoting=1)

    aggregatedFile = outputPath + "_aggregated.csv"
    
    # bodyFileDict = loadBodyDict(cleanedBodyData)
    print(">> Escribiendo resultados en ", aggregatedFile)
    with open(aggregatedFile, 'wb') as aggregatedData:
        fieldnames = ["Headline","ArticleBody", "Stance", "BodyIDS","BodyIDB"]
        
        writer = csv.DictWriter(aggregatedData, fieldnames=fieldnames)
        writer.writeheader()

        for index, line in cleanedStanceData.iterrows():
            # Buscamos el cuerpo asociado
            bodyId = line["Body ID"]
            associatedBody = cleanedBodyData.loc[cleanedBodyData['Body ID'] == bodyId, 'Body ID']
            if len(associatedBody) != 1:
                print(">> ERROR: He encontrado ", len(associatedBody), " elementos")
            else: 
                aggregatedLine = {
                    "Headline": line["Headline"],
                    "ArticleBody": associatedBody[0]["ArticleID"],
                    "Stance": line["Stance"],
                    "BodyIDS": bodyId,
                    "BodyIDB": associatedBody[0]["Body ID"],
                }
                # Escribimos la línea en el fichero
                writer.writerow(aggregatedLine)
