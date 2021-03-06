# Ejemplo de clasificador utilizando tensorflow de bajo nivel
# En este caso utilizamos la funcion dense() en vez de crear una funcion propia
import tensorflow as tf
import numpy as np
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import csv
import os
import time

EXECUTION_TAG = "_RESULTADOS_ARQUITECTURA_3_CAPAS_"

### Funciones auxiliares
#Vuelca las metricas de ejecucion 
def write_metrics_to_file(metrics):

    header = [
        # Arquitectura de la red e hiperparamentros de aprendizaje
        "activation_function","config_tag",
        "hidden_neurons", "n_layers",
        "optimizer", "learning_rate",
        "learning_decrease_base", "epochs",
        # Metricas
        "train_accuracy","test_accuracy",
        "precision_train","precision_test",
        "recall_train","recall_test",
        "confusion_matrix",
        # Regularizaciones
        "early_stopping_patience","dropout_rate","l2_regularization",
         "momentum_nesterov","execution_time", "execution_dir"
    ]
    
    csv_output_dir = "./executionStats/classifier/"
    date = time.strftime("%Y-%m-%d")
    output_file = csv_output_dir + '_FNN_classifier_' + date + '.csv'
    with open(output_file, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = header)
        newFile = os.stat(output_file).st_size == 0
        if newFile:
            writer.writeheader()
        writer.writerow(metrics)
        print(">> Stats exported to: ", output_file)
        print("############################################################################")

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
### Clasificador ###
default_hyperparams = {"activation_function": "relu", "learning_rate_update":"constant", "config_tag": "DEFAULT",
    "epochs": 20, 'hidden_neurons': [300, 100], "early_stopping": False, "learning_rate": 0.01, "dropout_rate": 1.0, "learning_decrease_base": 1, 
    "l2_scale": 0.0, 'batch_size': 50, "early_stopping_patience":2, "optimizer_function": "GD", "momentum": 0.9, "momentum_nesterov": True}
  
def modelClassifier(input_features, target, test_features, test_targets, hyperparams=default_hyperparams):
    print(">>> hyperparams: ", str(hyperparams))
    tf.reset_default_graph() 
    date = datetime.utcnow().strftime("%Y%m%d")
    hour = datetime.utcnow().strftime("%H%M%S")
    start = time.time()
    # root_logdir = "testLogs"
    execution_date = time.strftime("%m-%d")
    root_logdir = "fnnLogs/" + execution_date + EXECUTION_TAG
    tag = "FNNClassifier"
    config_tag = hyperparams.get("config_tag" , default_hyperparams["config_tag"])
    subdir = date
    logdir = "{}/{}/run-{}-{}-{}/".format(root_logdir, subdir , tag, config_tag, hour)
    print(">> LOG DIR: ", logdir)
    # Convertimos a enteros las clases
    train_labels = convert_to_int_classes(target)
    test_labels = convert_to_int_classes(test_targets)
    
    ### Definicion de la red ###
    train_samples = input_features.shape[0] # Numero de ejemplos
    
    ### Hiperparametros del modelo ###
    valid_hidden_neuron_param = ("hidden_neurons" in hyperparams and len(hyperparams["hidden_neurons"]) >= 2)
    hidden_neurons = hyperparams["hidden_neurons"] if valid_hidden_neuron_param else default_hyperparams["hidden_neurons"]
    early_stopping = hyperparams.get("early_stopping", default_hyperparams["early_stopping"])
    n_inputs = input_features.shape[1] #TamaÃ±o de la entrada
    
    # Numero de neuronas de la primera capa oculta
    n_hidden1 = hidden_neurons[0] 
    # Numero de neuronas de la segunda capa oculta
    n_hidden2 = hidden_neurons[1]
    n_layers = len(hidden_neurons)
    if n_layers >= 3:
        n_hidden3 = hidden_neurons[2]
    if n_layers >= 4:
        n_hidden4 = hidden_neurons[3]
    if n_layers >= 5:
        n_hidden5 = hidden_neurons[4]

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
    
    # Escala de regularizacion l2
    l2_scale =  hyperparams.get("l2_scale", default_hyperparams["l2_scale"])
    #Tasa de dropout
    drop_rate = hyperparams.get("dropout_rate", default_hyperparams["dropout_rate"])
    # Paciencia de early stopping
    early_stopping_patience = hyperparams.get("early_stopping_patience", default_hyperparams["early_stopping_patience"])
    # Funcion de optimizacion
    optimizer_function = hyperparams.get("optimizer_function", default_hyperparams["optimizer_function"])
    # Numero de epochs, tamano del batch, numero de iteraciones
    n_epochs = hyperparams["epochs"] if "epochs" in hyperparams else default_hyperparams["epochs"]
    batch_size = hyperparams.get("batch_size" , default_hyperparams["batch_size"])
    n_iterations = round(train_samples / batch_size)
    # Learning rate y esquema de actualizacion
    starter_learning_rate = hyperparams.get("learning_rate" , default_hyperparams["learning_rate"])
    learning_decrease_base = hyperparams.get("learning_decrease_base", default_hyperparams["learning_decrease_base"])
    
    momentum = hyperparams.get("momentum", default_hyperparams["momentum"])
    momentum_nesterov = ""
    
    print("> Shape de los datos de entrada (entrenamiento): ", input_features.shape)
    print("> Shape de los datos de entrada (test): ", test_features.shape)
    print("> Numero de neuronas de la capa de entrada: ", n_inputs)
    print("> Numero de instancias de entrenamiento: ", train_samples)
    print("> Funcion de activacion: ", hyperparams["activation_function"])
    print("> Numero de capas ocultas: ", n_layers)
    print(">> Numero de neuronas de las capas ocultas: ", str(hidden_neurons))
    print(">> Funcion optimizadora: ", optimizer_function)
    print(">> Momentum (aplicable si optimizacion momentum):", momentum)
    print("> Numero de epochs: ", n_epochs)
    print("> Learning rate: ", starter_learning_rate)
    print("> Learning rate base decay: ", learning_decrease_base)
    print("> Tam. batch: ", batch_size)
    print("> Prueba: ", config_tag)

    ### Network Architecture ###
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    keep_prob = tf.placeholder(tf.float32, shape=(None), name="keep_prob")
    
    if not drop_rate == 1.0:
        print(">> Dropout rate: ", drop_rate)
        with tf.name_scope("FNN"):
            # Definimos las capas ocultas
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation)
            dropout1 = tf.nn.dropout(hidden1, keep_prob, name="dropout_1_out")
            hidden2 = tf.layers.dense(dropout1, n_hidden2, name="hidden2", activation=activation)
            dropout2 = tf.nn.dropout(hidden2, keep_prob, name="dropout_2_out")
            if n_layers >= 3:
                hidden3 = tf.layers.dense(dropout2, n_hidden3, name="hidden3", activation=activation)
                dropout3 = tf.nn.dropout(hidden3, keep_prob, name="dropout_3_out")
            if n_layers >= 4:
                hidden4 = tf.layers.dense(dropout3, n_hidden4, name="hidden4", activation=activation)
                dropout4 = tf.nn.dropout(hidden4, keep_prob, name="dropout_4_out")
            if n_layers == 5:
                hidden5 = tf.layers.dense(dropout4, n_hidden5, name="hidden5", activation=activation)
                dropout5 = tf.nn.dropout(hidden5, keep_prob, name="dropout_5_out")
            
            
            # Definimos la capa de salida
            if n_layers == 3:
                logits = tf.layers.dense(dropout3, n_outputs, name="outputs")
            elif n_layers == 4:
                logits = tf.layers.dense(dropout4, n_outputs, name="outputs")
            elif n_layers == 5:
                logits = tf.layers.dense(dropout5, n_outputs, name="outputs")
            else:
                logits = tf.layers.dense(dropout2, n_outputs, name="outputs")

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            final_loss = tf.reduce_mean(xentropy, name="loss")
    else:
        print(">> L2 SCALE: ", l2_scale)
        with tf.name_scope("FNN"):
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale) if l2_scale >= 0.0 else None
            #with tf.contrib.framework.arg_scope([tf.layers.dense], kernel_regularizer=l2_regularizer):    
            # Definimos las capas ocultas
            hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation, kernel_regularizer=l2_regularizer)
            hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation, kernel_regularizer=l2_regularizer)
            if n_layers >= 3:
                hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=activation, kernel_regularizer=l2_regularizer)
            if n_layers >= 4:
                hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=activation, kernel_regularizer=l2_regularizer)
            if n_layers == 5:
                hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5", activation=activation, kernel_regularizer=l2_regularizer)
            
            
            # Definimos la capa de salida
            if n_layers == 3:
                logits = tf.layers.dense(hidden3, n_outputs, name="outputs", kernel_regularizer=l2_regularizer)
            elif n_layers == 4:
                logits = tf.layers.dense(hidden4, n_outputs, name="outputs", kernel_regularizer=l2_regularizer)
            elif n_layers == 5:
                logits = tf.layers.dense(hidden5, n_outputs, name="outputs", kernel_regularizer=l2_regularizer)
            else:
                logits = tf.layers.dense(hidden2, n_outputs, name="outputs", kernel_regularizer=l2_regularizer)

        # We define the cost function
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            if l2_scale >= 0.0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                loss = tf.reduce_mean(xentropy, name="base_loss")
                final_loss = tf.add_n([loss] + reg_losses, name="loss")
            else:
                final_loss = tf.reduce_mean(xentropy, name="loss")

    ### Training ### 
    if learning_decrease_base == 1:
        learning_rate = starter_learning_rate
    else:
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, n_iterations, learning_decrease_base)
    
    with tf.name_scope("train"):
        if optimizer_function == 'ADAM':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_function == 'momentum':
            momentum_nesterov = hyperparams.get("momentum_nesterov", default_hyperparams["momentum_nesterov"])
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=momentum_nesterov)
        else: # GD by default
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # training_op = optimizer.minimize(loss)
        if learning_decrease_base == 1:
            training_op = optimizer.minimize(final_loss)
        else:
            training_op = optimizer.minimize(final_loss, global_step = global_step)

    ### Metrics and results ###
    # with tf.name_scope("eval"):
    correct_prediction = tf.nn.in_top_k(logits, y , 1)
    prediction=tf.argmax(logits,1,name="prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # recall = tf.metrics.recall(y, prediction)
    # precision = tf.metrics.precision(y, prediction)
    confusion_matrix_class = tf.confusion_matrix(y, prediction)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
        
    
    ### Export de escalares e histogramas ###
    # Sacamos el valor actual de los dos accuracy en los logs para visualizarlo en tensorboard
    # Descomentar los que resulten interesantes de visualizar
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Loss', final_loss)
    tf.summary.histogram('Xentropy', xentropy)
    tf.summary.scalar('Learning Rate', learning_rate)
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
    
    
    #### Fase de ejecucion ###
    # Entrenamos el modelo. Usamos minibatch gradient descent 
    # (en cada iteracion aplicamos el gradiente descendiente sobre una submuestra aleatoria de los datos de entrenamiento)
    #  Al final de cada epoch computamos el accuracy sobre uno de los batches.
    with tf.Session() as sess:
        # Inicializamos las variables globales del grafo
        init.run()
        # Creamos el writter
        summary_writer = tf.summary.FileWriter(logdir + '_train', tf.get_default_graph())
        summary_writer_test = tf.summary.FileWriter(logdir + '_test', tf.get_default_graph())
        # Realizamos el entrenamiento fijando en n_epochs
        minimun_loss = 1000 #Arbitrary initial value
        early_stopping_threshold = round(n_iterations * early_stopping_patience)
        executed_epochs = 0
        stop_training = False
        loss_stacionality = 0
        logits_results = None
        for epoch in range(n_epochs):
            #print(">> Epoch ", epoch)
            #print("tf.graphKeys: ", str(tf.GraphKeys))
            if not stop_training or not early_stopping: 
                for iteration in range(n_iterations):
                    # X_batch, y_batch = mnist.train.next_batch(batch_size)
                    X_batch, y_batch = next_batch(batch_size, input_features, train_labels)
                    _, summary = sess.run([training_op, merged_summary_op], feed_dict={X: X_batch, y: y_batch, keep_prob: drop_rate})
                    #Comprobamos si el modelo va mejorando con respecto a los datos de entrenamiento
                    if iteration % 20 ==  0:
                     loss_test = sess.run(final_loss, feed_dict={X: test_features, y: test_labels, keep_prob: drop_rate})
                     if loss_test < minimun_loss and early_stopping:
                         minimun_loss = loss_test
                         loss_stacionality = 0
                         # Guardamos el estado de la red
                         save_path = saver.save(sess, logdir + "/model/my_model_final.ckpt")
                     else:
                         loss_stacionality = loss_stacionality + 1
                             
                # Obtenemos el accuracy de los datos de entrenamiento y los de tests    
                # acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                #acc_train, summary_train, r_losses, b_loss, f_loss = sess.run([accuracy, merged_summary_op,reg_losses, loss, final_loss], feed_dict={X: input_features, y: train_labels, keep_prob: drop_rate})
                acc_train, summary_train = sess.run([accuracy, merged_summary_op], feed_dict={X: input_features, y: train_labels, keep_prob: drop_rate})
                acc_test, summary_test = sess.run([accuracy, merged_summary_op], feed_dict={X: test_features, y: test_labels, keep_prob: 1.0})
                
                summary_writer.add_summary(summary_train, epoch * n_iterations)        
                summary_writer.flush() 
                summary_writer_test.add_summary(summary_test, epoch * n_iterations)        
                summary_writer_test.flush() 
                executed_epochs = executed_epochs + 1
                
                stop_training = loss_stacionality * 20 >= early_stopping_threshold
                #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                print(epoch, "Train accuracy: ", acc_train, " Test accuracy: ", acc_test)
                #print("> BASE LOSS: ", str(b_loss))
                #print("> REG LOSSES: ", str(r_losses))
                #print("> FINAL LOSS: ", str(f_loss))
        end = time.time()
        # Ejecutamos las metricas finales
        sess.run(tf.local_variables_initializer())
        acc_final_train = sess.run(accuracy, feed_dict={X: input_features, y: train_labels, keep_prob: drop_rate})
        prediction_values_train = sess.run(prediction, feed_dict={X: input_features, y: train_labels, keep_prob: drop_rate})

        acc_final_test, logits_results = sess.run([accuracy, logits], feed_dict={X: test_features, y: test_labels, keep_prob: 1.0})
        prediction_values = sess.run(prediction, feed_dict={X: test_features, y: test_labels, keep_prob: 1.0})
        # Obtenemos el score del clasificador, para pintar la curva ROC
        
        y_score = np.array(logits_results)
        
        # logits_test = sess.run(logits,feed_dict={X: test_features, y: test_labels, keep_prob: 1.0} )
        #accuracy_prediction = sess.run(accuracy_prediction,feed_dict={X: test_features, y: test_labels} )
        #print("Prediction: ", prediction_values)
        #print("Longitud de predictions: ", sess.run(tf.size(prediction_values)))
        #print("Longitud de las entradas: ", len(test_labels))
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
        
        if not early_stopping:
            # Guardamos la version final del modelo entrenado
            save_path = saver.save(sess, logdir + "/model/my_model_final.ckpt")
        
        
        metrics = {
            "activation_function": hyperparams["activation_function"],
            "config_tag": config_tag,
            
            "hidden_neurons": hidden_neurons,
            "n_layers": n_layers,
            "optimizer": optimizer_function,
            "learning_rate": str(starter_learning_rate),
            "learning_decrease_base": learning_decrease_base,
            "epochs": executed_epochs,

            "train_accuracy": round(acc_final_train,2),
            "test_accuracy": round(acc_final_test,2),
            "precision_train":round(precision_train,2),
            "precision_test": round(precision_classSK,2),
            "recall_train": round(recall_train,2),
            "recall_test": round(recall_classSK,2),
            "confusion_matrix": confusion_matrix_class,
            
            "early_stopping_patience": early_stopping_patience,
            "dropout_rate": drop_rate,
            "l2_regularization": round(l2_scale,3),
            "momentum_nesterov": momentum_nesterov,

            "execution_time": end - start,
            "execution_dir": logdir      
		    }
        print(">> MLP Metrics: ")
        print(metrics)
        write_metrics_to_file(metrics)
        #Anadimos dos campos extra, que no se visualizan en el csv
        metrics["y_true"]= test_labels
        metrics["y_score"] = y_score
        return metrics
