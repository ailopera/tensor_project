import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test)


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     print(type(cm))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# # Compute confusion matrix
# # cnf_matrix = confusion_matrix(y_test, y_pred)
# # np.set_printoptions(precision=2)

# # Read data Results
executionStatsPath = 'executionStats/' + "bag_Of_Words_execution_2018-04-20.csv"
# executionStatsPath = 'executionStats/' + "vector_Average_execution_2018-04-18.csv"
trainExecution = pd.read_csv(executionStatsPath,header=0,delimiter=",", quoting=1)
print("> Load file ", executionStatsPath)
class_names = ['Agree', 'Disagree', 'Discuss', 'Unrelated']


# for index, execution in trainExecution.iterrows():
#     # print(execution)
#     cnf_matrix = np.array(execution['confusionMatrix'])    
#     # print(type(cnf_matrix))
#     # Plot non-normalized confusion matrix
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=class_names,
#                         title='Confusion matrix, without normalization')

#     # Plot normalized confusion matrix
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                         title='Normalized confusion matrix')

#     # plt.show()
#     fname = 'executionStats/plots/' + execution['date'] + "_" + execution['executionDesc'] + "_"  + execution['modelTrained'] + "_"  + execution['modelName']
#     plt.savefig(fname)
#     print("Plot saved as ", fname)



# for index, execution in trainExecution.iterrows():
#     cnf_matrix = np.array(execution['confusionMatrix'])    
#     plot, ax = plot_confusion_matrix(conf_mat=cnf_matrix)
#     # plt.show()
#     fname = 'executionStats/plots/' + execution['date'] + "_" + execution['executionDesc'] + "_"  + execution['modelTrained'] + "_"  + execution['modelName']
#     plt.savefig(fname)
#     print("Plot saved as ", fname)

    

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import ast
import json
import re
# array = [[33,2,0,0,0,0,0,0,0,1,3], 
#         [3,31,0,0,0,0,0,0,0,0,0], 
#         [0,4,41,0,0,0,0,0,0,0,1], 
#         [0,1,0,30,0,6,0,0,0,0,1], 
#         [0,0,0,0,38,10,0,0,0,0,0], 
#         [0,0,0,3,1,39,0,0,0,0,4], 
#         [0,2,2,0,4,1,31,0,0,0,2],
#         [0,1,0,0,0,0,0,36,0,2,0], 
#         [0,0,0,0,0,0,1,5,37,5,1], 
#         [3,0,0,0,0,0,0,0,0,39,0], 
#         [0,0,0,0,0,0,0,0,0,0,38]]

    
for index, execution in trainExecution.iterrows():
    # print(type(execution['confusionMatrix']))
    # mat = execution['confusionMatrix'].replace('\s*', ',')
    mat = re.sub("\s+", ",", execution['confusionMatrix'].strip())
    mat = mat.replace("[,", "[")
    print(mat)
    cnf_matrix = ast.literal_eval(mat)
    # json.loads(execution['confusionMatrix'])
    # cnf_matrix = np.array(execution['confusionMatrix'])    
    # cnf_matrix = execution['confusionMatrix']
    # print(cnf_matrix.shape)
    # plot, ax = plot_confusion_matrix(conf_mat=cnf_matrix)
    
    df_cm = pd.DataFrame(cnf_matrix, index = [0,1,2,3],
                  columns = [0,1,2,3])
    plt.figure(figsize = (10,7))
    sn_plot = sn.heatmap(df_cm, annot=True,fmt="d").get_figure()
    
    # plt.show()
    fname = execution['executionDesc'] + "_" + execution['modelTrained'] + "_" + "row" + str(index)
    # fname = str(execution['executionDesc']) + "_"  + str(execution['modelTrained']) + "_"  + str(execution['modelName']) 
    sn_plot.savefig(fname)
    print("Plot saved as ", fname)