import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#  Utilidades varias
n_classes = 4

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
    filename = label + '.png'
    fig.savefig(filename, bbox_inches='tight') 

# Pinta las curvas ROC para los datos de test
def defineROCCurves(y_test, y_score, execution_label):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    colors = ['darkorange', 'darkorange', 'darkorange', 'darkorange']    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        label = execution_label+str(i)
        plotROCCurves(fpr[i], tpr[i], roc_auc[i], colors[i], label)
    
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

