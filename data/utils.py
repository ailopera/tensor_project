import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#  Utilidades varias
n_classes = 3 # Clases: 0, 1, 2, 3


def plotROCCurves(fpr, tpr, roc_auc, color, label):
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color=color,
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + label)
    plt.legend(loc="lower right")
    # plt.show()
    fig = plt.figure()
    filename = "prueba" + '.png'
    fig.savefig(filename, bbox_inches='tight') 

# Pinta las curvas ROC para los datos de test
def defineROCCurves(y_test, y_score, execution_label):
    
    print(">>> Execution label: ", execution_label)
    print(">>> Y_test shape: ", y_test.shape)
    print(">>> Y_Score shape: ", y_score.shape)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    colors = ['darkorange', 'darkorange', 'darkorange', 'darkorange']    
    for i in range(n_classes):
        pos_label = i
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label)
        roc_auc[i] = auc(fpr[i], tpr[i])
        label = execution_label+str(i)
        plotROCCurves(fpr[i], tpr[i], roc_auc[i], colors[i], label)
    
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

