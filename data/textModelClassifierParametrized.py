# Ejemplo de clasificador utilizando tensorflow de bajo nivel
# En este caso utilizamos la funcion dense() en vez de crear una funcion propia
import tensorflow as tf
import numpy as np
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import csv
import os

### Funciones auxiliares
#Vuelca las metricas de ejecucion 
def write_metrics_to_file(metrics):
    header = ["train_accuracy", "test_accuracy", "confusion_matrix",
		"precision_train", "recall_train", "precision_test",
        "recall_test" ,"execution_dir","activation_function"
    ]
    csv_output_dir = "./executionStats/classifier"
    date = time.strftime("%Y-%m-%d")
    output_file = csv_output_dir + '_' + date + '_FNN_classifier.csv'
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldNames = header)
        newFile = os.stat(output_file).st_size == 0
        if newFile:
            writer.writeheader()
        writer.writerow(executionData)
        print(">> Stats exported to: ", output_file)


# Toma N muestras de forma aleatoria a partir de los datos de entrada 
def next_batch(batch_size, train_data, target_data):
    training_shape = train_data.shape[0]
    minibatch_indexes = random.sample(range(0,training_shape), batch_size)
    # print("> Minibatch_indexes: ", len(minibatch_indexes))
    # Tomamos las muestras de los datos de entrada
    minibatch_data = []
    minibatch_targets = []
    for index in minibatch_indexes:
        sample = train_data[index]
        sample_target = target_data[index]
        minibatch_data.append(sample)
        minibatch_targets.append(sample_target)

    #print("> Len(minibatch_data): ", len(minibatch_data))
    #print("> Len(minibatch_targets): ", len(minibatch_targets))
    #print("> Data sample: ", minibatch_data[0])
    #print("> Target sample: ", minibatch_targets)
    #print("-------------")
    return np.array(minibatch_data),np.array(minibatch_targets)


def convert_to_int_classes(targetList):
    map = {
        "agree": 0,
        "disagree": 1,
        "discuss": 2,
        "unrelated": 3
    }
    int_classes = []
    for elem in targetList:
        int_classes.append(map[elem])
        
    return np.array(int_classes)


# activation_function: relu | leaky_relu | elu | 
# learning_rate_update: constant | step_decay | exponential_decay
### Clasificador ###
default_hyperparams = {"activation_function": "relu", "learning_rate_update":"constant", "config_tag": "DEFAULT",
    "hidden1": 300 , "hidden2": 100}

def modelClassifier(input_features, target, test_features, test_targets, hyperparams=None):
    tf.reset_default_graph() 
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "testLogs"
    tag = "FNNClassifier"
    config_tag = hyperparams["config_tag"] if not hyperparams == None else default_hyperparams["config_tag"]
    logdir = "{}/run-{}-{}-{}/".format(root_logdir,tag, config_tag,now)
      
    # Convertimos a enteros las clases
    train_labels = convert_to_int_classes(target)
    test_labels = convert_to_int_classes(test_targets)
    
    ### Definicion de la red ###
    train_samples = input_features.shape[0] # Numero de ejemplos

    # Hiperparametros del modelo
    n_inputs = input_features.shape[1] #TamaÃ±o de la entrada
    # Numero de neuronas de la primera capa oculta
    n_hidden1 = hyperparams['hidden1'] if hyperparams['hidden1'] is not None else default_hyperparams['hidden1']
    # Numero de neuronas de la segunda capa oculta
    n_hidden2 = hyperparams['hidden2'] if hyperparams['hidden2'] is not None else default_hyperparams['hidden2']
    n_outputs = 4 # Numero de salidas/clases a predecir

    # funcion de activacion de las capas
    activation = None
    if hyperparams["activation_function"] == 'relu':
        activation = tf.nn.relu
    elif hyperparams["activation_function"] == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif hyperparams["activation_function"] == 'elu':
        activation = tf.nn.elu
    else:
        print(">>> ERROR: Wrong activation function specified")
        return


    print("> Shape de los datos de entrada (entrenamiento): ", input_features.shape)
    print("> Shape de los datos de entrada (test): ", test_features.shape)
    print("> Numero de neuronas de la capa de entrada: ", n_inputs)
    print("> Numero de instancias de entrenamiento: ", train_samples)

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation)
        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    # We define the cost function
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    # Definimos el entrenamiento 
    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    # Definimos las metricas
    # with tf.name_scope("eval"):
    correct_prediction = tf.nn.in_top_k(logits, y , 1)
    prediction=tf.argmax(logits,1)
    #correct_prediction_2 = tf.equal(prediction, y)
    #accuracy_prediction = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    recall = tf.metrics.recall(y, prediction)
    precision = tf.metrics.precision(y, prediction)
    confusion_matrix_class = tf.confusion_matrix(y, prediction)
    #final_accuracy = tf.metrics.accuracy(y, prediction)
    #con_mat = tf.confusion_matrix(labels=y, predictions=prediction, num_classes=4, dtype=tf.int32, name=None)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    ##Metricas de scikit
    #precision_classSK = precision_score(y, prediction, average="weighted", labels=[0,1,2,3])
    #recall_classSK = recall_score(y, prediction, average="weighted", labels=[0,1,2,3])
            
    #### Fase de ejecucion ###
    n_epochs = 20
    batch_size = 50

    n_iterations = round(train_samples / batch_size)

    # Export de escalares
    # Sacamos el valor actual de los dos accuracy en los logs para visualizarlo en tensorboard
    
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Loss', loss)
    tf.summary.histogram('Xentropy', xentropy)
    # hidden_1_weights = [v for v in tf.global_variables() if v.name == "hidden1/kernel:0"][0]
    # hidden_2_weights = [v for v in tf.global_variables() if v.name == "hidden2/kernel:0"][0]
    # outputs_weigths = [v for v in tf.global_variables() if v.name == "outputs/kernel:0"][0]
    # hidden_1_bias = [v for v in tf.global_variables() if v.name == "hidden1/bias:0"][0]
    # hidden_2_bias = [v for v in tf.global_variables() if v.name == "hidden2/bias:0"][0]
    # output_2_bias = [v for v in tf.global_variables() if v.name == "outputs/bias:0"][0]
    # tf.summary.histogram('Weigths_hidden1', hidden_1_weights)
    # tf.summary.histogram('Weigths_hidden2', hidden_2_weights)
    # tf.summary.histogram('Weigths_hidden2', hidden_2_weights)
    # tf.summary.histogram('Bias_hidden1', hidden_1_bias)
    # tf.summary.histogram('Bias_hidden2', hidden_2_bias)
    # tf.summary.histogram('Bias_output', output_2_bias)
    # #tf.summary.scalar('Precision', precision_classSK)
    #tf.summary.scalar('Recall', recall_classSK)
    merged_summary_op = tf.summary.merge_all()
    
    # Entrenamos el modelo. Usamos minibatch gradient descent 
    # (en cada iteracion aplicamos el gradiente descendiente sobre una submuestra aleatoria de los datos de entrenamiento)
    #  Al final de cada epoch computamos el accuracy sobre uno de los batches.
    with tf.Session() as sess:
        # for v in tf.global_variables():
        #   print(v.name)
        # # hidden_1_weights = [v for v in tf.global_variables() if v.name == "hidden1/kernel:0"][0]
        
        # Inicializamos las variables globales del grafo
        init.run()
        # Creamos el writter
        summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
        # Realizamos el entrenamiento fijando en n_epochs
        for epoch in range(n_epochs):
            for iteration in range(n_iterations):
                # X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch, y_batch = next_batch(batch_size, input_features, train_labels)
                _, summary = sess.run([training_op, merged_summary_op], feed_dict={X: X_batch, y: y_batch})
                #Escribimos las metricas en tensorboard
                if iteration % 20 ==  0:
                    summary_writer.add_summary(summary, epoch * n_iterations + iteration)        
            # Obtenemos el accuracy de los datos de entrenamiento y los de tests    
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            #acc_train, h1_weights = sess.run([accuracy, hidden_1_weights], feed_dict={X: X_batch, y: y_batch})
            #acc_test = accuracy.eval(feed_dict={X: test_features, y: test_labels})
            acc_test, prediction_values = sess.run([accuracy, prediction], feed_dict={X: test_features, y: test_labels})
            
            summary_writer.flush() 
            print(epoch, "Train accuracy: ", acc_train, " Test accuracy: ", acc_test)
            
        
        # Ejecutamos las metricas finales
        sess.run(tf.local_variables_initializer())
        acc_final_train = sess.run(accuracy, feed_dict={X: input_features, y: train_labels})
        acc_final_test = sess.run(accuracy, feed_dict={X: test_features, y: test_labels})
        # recall_class = sess.run(recall, feed_dict={X: test_features, y: test_labels})
        # precision_class = sess.run(precision, feed_dict={X: test_features, y: test_labels})
        # confusion_matrix_class = sess.run(confusion_matrix_class, feed_dict={X: test_features, y: test_labels})
        prediction_values = sess.run(prediction, feed_dict={X: test_features, y: test_labels})
        prediction_values_train = sess.run(prediction, feed_dict={X: input_features, y: train_labels})

        logits = sess.run(logits,feed_dict={X: test_features, y: test_labels} )
        
        #accuracy_prediction = sess.run(accuracy_prediction,feed_dict={X: test_features, y: test_labels} )
        print("Prediction: ", prediction_values)
        print("Longitud de predictions: ", sess.run(tf.size(prediction_values)))
        print("Longitud de las entradas: ", len(test_labels))
        #sess.run(tf.Print(prediction_values, [prediction_values]))
        #print("logits: ", logits)
        #print("Y: ", y)
        #print("Acuraccy prediction: ", accuracy_prediction)
        #confusion_matrix_class = confusion_matrix(test_labels,prediction)
        
        # Calculamos precision, recall y confusion matrix utilizando sklearn
        precision_train = precision_score(train_labels, prediction_values_train, average="weighted", labels=[0,1,2,3])
        recall_train = recall_score(train_labels, prediction_values_train, average="weighted", labels=[0,1,2,3])
        
        confusion_matrix_class = confusion_matrix(test_labels, prediction_values,labels=[0,1,2,3])
        precision_classSK = precision_score(test_labels, prediction_values, average="weighted", labels=[0,1,2,3])
        recall_classSK = recall_score(test_labels, prediction_values, average="weighted", labels=[0,1,2,3])
        # print("Tipo: ", type(precision_class))
        # print("valor: ", precision_class)
        # Guardamos la version final del modelo entrenado
        save_path = saver.save(sess, logdir + "/my_model_final.ckpt")
        
        
        metrics = {
            "train_accuracy": round(acc_final_train,2),
            "test_accuracy": round(acc_final_test,2),
            "confusion_matrix": confusion_matrix_class,
            "precision_train":round(precision_train,2),
            "recall_train": round(recall_train,2),
            "precision_test": round(precision_classSK,2),
            "recall_test": round(recall_classSK,2),
            "execution_dir": logdir,
            "activation_function": hyperparams["activation_function"],
            "hidden1": hyperparams["hidden1"],
            "hidden2": hyperparams["hidden2"],
		}
        print(">> MLP Metrics: ")
        print(metrics)
        write_metrics_to_file(metrics)
        return metrics
