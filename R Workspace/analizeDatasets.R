library(ggplot2)
library(gridExtra)
setwd("C:/Users/Operador/projects/tensor_project/R Workspace")
# Read in csv files
# Datasets originales
#dfTrain <- read.table("data/train_data_aggregated.csv", header = TRUE, sep = ",")
#dfTest <- read.table("data/test_data_aggregated.csv", header = TRUE, sep = ",")
#dfCompetition <- read.table("data/competition_data_aggregated.csv", header = TRUE, sep = ",")

#Particionado 1
#dfTrain <- read.table("../data/fnc-1-original/finalDatasets/train_partition.csv", header = TRUE, sep = ",")
#dfTest <- read.table("../data/fnc-1-original/finalDatasets/test_partition_3.csv", header = TRUE, sep = ",")
# Particionado 2
#dfTrain <- read.table("../data/fnc-1-original/finalDatasets/train_partition_split.csv", header = TRUE, sep = ",")
dfTrain <- read.table("../data/executionStats/bag_Of_Words_smoteData_2018-06-20_smote_all.csv", header = TRUE, sep = ",")
dfTest <- read.table("../data/fnc-1-original/finalDatasets/test_partition_split.csv", header = TRUE, sep = ",")

# Particionado 3
#dfTrain <- read.table("../data/fnc-1-original/finalDatasets/train_partition_split.csv", header = TRUE, sep = ",")
#dfTest <- read.table("../data/fnc-1-original/finalDatasets/train_partition_3.csv", header = TRUE, sep = ",")

# head(df$Stance)


# Barplots de conteo de clasificaciones
# TODO: Añadir porcentaje en cada barplot
# TODO: Explicar en la memoria estos gráficos
plotTrain <- ggplot(data=dfTrain, aes(x=Stance)) +  geom_bar(stat="count", fill="steelblue") + labs(title="Clasificaciones en el conjunto de entrenamiento/validación", 
       x="Etiqueta", y="Frecuencia")+ stat_count( aes(y=..count../sum(..count..), label=..count..), geom="text")

plotTest <- ggplot(data=dfTest, aes(x=Stance)) +  geom_bar(stat="count", fill="steelblue") + labs(title="Clasificaciones en el conjunto de test (Particionado 2)", 
       x="Etiqueta", y="Frecuencia") + stat_count( aes(y=..count../sum(..count..), label=..count..), geom="text")


plotTrain
plotTest
