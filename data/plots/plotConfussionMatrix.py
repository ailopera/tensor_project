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
# executionStatsPath = '../executionStats/' + "bag_Of_Words_execution_2018-04-21.csv"
executionStatsPath = '../executionStats/' + "vector_Average_execution_2018-04-21.csv"
trainExecution = pd.read_csv(executionStatsPath,header=0,delimiter=",", quoting=1)
print("> Load file ", executionStatsPath)
class_names = ['Agree', 'Disagree', 'Discuss', 'Unrelated']

    
for index, execution in trainExecution.iterrows():
    # print(type(execution['confusionMatrix']))
    # mat = execution['confusionMatrix'].replace('\s*', ',')
    mat = re.sub("\s+", ",", execution['confusionMatrix'].strip())
    mat = mat.replace("[,", "[")
    print(mat)
    cnf_matrix = ast.literal_eval(mat)
    
    df_cm = pd.DataFrame(cnf_matrix, index = class_names,
                  columns = class_names)
    plt.figure(figsize = (5,4))
    sn_plot = sn.heatmap(df_cm, annot=True,fmt="d").get_figure()
    
    # plt.show()
    fname = execution['executionDesc'] + "_" + execution['modelTrained'] + "_" + "row" + str(index+2)
    # fname = str(execution['executionDesc']) + "_"  + str(execution['modelTrained']) + "_"  + str(execution['modelName']) 
    sn_plot.savefig(fname)
    print("Plot saved as ", fname)