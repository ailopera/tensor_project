import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
import ast
import json
import re

# # Read data Results
# Resultados de modelos de representación de textos
# Ficheros del particionado 1
# executionStatsPath = '../executionStats/' + "bag_Of_Words_execution_2018-04-23.csv"
# executionStatsPath = '../executionStats/' + "vector_Average_execution_2018-04-23.csv"
# Ficheros del particionado 2
# executionStatsPath = '../executionStats/' + "bag_Of_Words_execution_2018-05-01.csv"
# executionStatsPath = '../executionStats/' + "bag_Of_Words_execution_2018-05-02_smote_all.csv"
# executionStatsPath = '../executionStats/' + "vector_Average_execution_2018-05-03.csv"
# executionStatsPath = '../executionStats/' + "vector_Average_execution_2018-05-01.csv"

# Resultados FNN, Particionado 2
executionStatsPath = '../executionStats/classifier/' + "_FNN_classifier_2018-05-26.csv"
trainExecution = pd.read_csv(executionStatsPath,header=0,delimiter=",", quoting=1)
print("> Load file ", executionStatsPath)
class_names = ['Agree', 'Disagree', 'Discuss', 'Unrelated']

    
for index, execution in trainExecution.iterrows():
    # print(type(execution['confusionMatrix']))
    # mat = re.sub("\s+", ",", execution['confusionMatrix'].strip()) #Para csv de modelos de representacion
    mat = re.sub("\s+", ",", execution['confusion_matrix'].strip()) # Para csv de clasificador neuronal
    mat = mat.replace("[,", "[")
    print(mat)
    cnf_matrix = ast.literal_eval(mat)
    
    df_cm = pd.DataFrame(cnf_matrix, index = class_names,
                  columns = class_names)
    plt.figure(figsize = (5,4))

    # Para cambiar el esquema de color:
    # plot using a color palette
    # sns.heatmap(df, cmap="YlGnBu")
    # sns.heatmap(df, cmap="Blues")
    # sns.heatmap(df, cmap="BuPu")
    # sns.heatmap(df, cmap="Greens")

    sn_plot = sn.heatmap(df_cm, annot=True,fmt="d", cmap="Blues").get_figure()
    
    # plt.show()
    # fname = execution['executionDesc'] + "_" + execution['modelTrained'] + "_" + "row" + str(index+2) # Para csv de modelos de representacion
    fname = "row" + str(index+2) # Para csv de clasificador neuronal
    # fname = str(execution['executionDesc']) + "_"  + str(execution['modelTrained']) + "_"  + str(execution['modelName']) 
    sn_plot.savefig(fname)
    print("Plot saved as ", fname)