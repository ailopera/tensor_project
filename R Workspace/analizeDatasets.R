library(ggplot2)
library(gridExtra)

# Read in csv files
dfTrain <- read.table("data/train_data_aggregated.csv", header = TRUE, sep = ",")
dfTest <- read.table("data/test_data_aggregated.csv", header = TRUE, sep = ",")
dfCompetition <- read.table("data/competition_data_aggregated.csv", header = TRUE, sep = ",")

# head(df$Stance)

# Barplots de conteo de clasificaciones
# TODO: Añadir porcentaje en cada barplot
# TODO: Explicar en la memoria estos gráficos
plotTrain <- ggplot(data=dfTrain, aes(x=Stance)) +  geom_bar(stat="count", fill="steelblue") + labs(title="Clasificaciones en el conjunto de entrenamiento", 
       x="Etiqueta", y="Frecuencia")

plotTest <- ggplot(data=dfTest, aes(x=Stance)) +  geom_bar(stat="count", fill="steelblue") + labs(title="Clasificaciones en el conjunto de test", 
       x="Etiqueta", y="Frecuencia")

plotCompetition <- ggplot(data=dfCompetition, aes(x=Stance)) +  geom_bar(stat="count", fill="steelblue") + labs(title="Clasificaciones en el conjunto de validación", 
                                                                                                  x="Etiqueta", y="Frecuencia")

plotCompetition