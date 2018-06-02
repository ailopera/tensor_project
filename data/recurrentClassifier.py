import tensorflow as tf
import numpy as np
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import csv
import os
import time

EXECUTION_TAG = "prueba_2"

### Funciones auxiliares
def write_metrics_to_file(metrics):
    header = ["config_tag", "train_accuracy", "test_accuracy", "confusion_matrix",
		"precision_train", "precision_test",
        "recall_train", "recall_test" ,
        "execution_dir", "execution_time"
    ]
    csv_output_dir = "./executionStats/classifier/RNN"
    date = time.strftime("%Y-%m-%d")
    output_file = csv_output_dir + '_RNN_classifier_' + date + '.csv'
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


### Clasificador ###
default_hyperparams = { "arquitecture": "simple", "config_tag": "default"  }
def modelClassifier(input_features, target, test_features, test_targets, hyperparams=None):
    print(">>> hyperparams: ", str(hyperparams))
    tf.reset_default_graph() 
    date = datetime.utcnow().strftime("%Y%m%d")
    hour = datetime.utcnow().strftime("%H%M%S")
    start = time.time()
    # root_logdir = "testLogs"
    execution_date = time.strftime("%m-%d")
    root_logdir = "RNNLogs/" + execution_date + EXECUTION_TAG
    tag = "RNNClassifier"
    #config_tag = "Prueba"
    config_tag = hyperparams.get("config_tag" , default_hyperparams["config_tag"])
    subdir = date
    logdir = "{}/{}/run-{}-{}-{}/".format(root_logdir, subdir , tag, config_tag, hour)

    # Convertimos a enteros las clases
    train_labels = convert_to_int_classes(target)
    test_labels = convert_to_int_classes(test_targets)

    print(">> Input shape: ", input_features.shape)
    arquitecture = hyperparams.get("arquitecture", default_hyperparams["arquitecture"])
    n_steps = 2
    n_inputs = 300
    
    n_neurons = 150
    n_outputs = 4

    # Redimensionamos los datos de test
    test_features = test_features.reshape((-1, n_steps, n_inputs))
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    if arquitecture == "simple":
        with tf.name_scope("rnn"):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
            # lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
            outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
            logits = tf.layers.dense(states, n_outputs)
    elif arquitecture == "multi":
        n_layers = 3
        with tf.name_scope("rnn"):
            layers = [tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
                    for layer in range(n_layers)]
            #layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
            #        for layer in range(n_layers)]
            multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
            outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)


    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)

    learning_rate = 0.001
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("metrics"):
        correct = tf.nn.in_top_k(logits, y, 1)
        prediction=tf.argmax(logits,1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 150
    
    train_samples = input_features.shape[0] # Numero de ejemplos
    
    n_iterations = round(train_samples / batch_size)
    print("> Numero de instancias de entrenamiento: ", train_samples)
    print("> Numero de epochs: ", n_epochs)
    print("> Learning rate: ", learning_rate)
    print("> Tam. batch: ", batch_size)
    print("> Prueba: ", config_tag)

    #Definimos los escalares que visualizaremos
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Loss', loss)
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        init.run()
        # Creamos el writter
        summary_writer = tf.summary.FileWriter(logdir + '_train', tf.get_default_graph())
        summary_writer_test = tf.summary.FileWriter(logdir + '_test', tf.get_default_graph())
        for epoch in range(n_epochs):
            for iteration in range(n_iterations):
                X_batch, y_batch = next_batch(batch_size, input_features, train_labels)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_features, y: test_labels})
            
            # Redimensionamos los datos de test
            train_features = input_features.reshape((-1, n_steps, n_inputs))
            acc_train, summary_train = sess.run([accuracy, merged_summary_op], feed_dict={X: train_features, y: train_labels})
            acc_test, summary_test = sess.run([accuracy, merged_summary_op], feed_dict={X: test_features, y: test_labels})

            summary_writer.add_summary(summary_train, epoch * n_iterations)        
            summary_writer.flush() 
            summary_writer_test.add_summary(summary_test, epoch * n_iterations)        
            summary_writer_test.flush() 

            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            # Guardamos la version actual del modelo entrenado
            save_path = saver.save(sess, "./my_model_final.ckpt")
            # Imprimimos el grafo para verlo desde tensorflow
            file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
            
        end = time.time()
        # Ejecutamos las metricas finales
        sess.run(tf.local_variables_initializer())
        acc_final_train = sess.run(accuracy, feed_dict={X: train_features, y: train_labels})
        prediction_values_train = sess.run(prediction, feed_dict={X: train_features, y: train_labels})

        acc_final_test = sess.run(accuracy, feed_dict={X: test_features, y: test_labels})
        prediction_values = sess.run(prediction, feed_dict={X: test_features, y: test_labels})
       
        # Calculamos precision, recall y confusion matrix utilizando sklearn
        precision_train = precision_score(train_labels, prediction_values_train, average="weighted", labels=[0,1,2,3])
        recall_train = recall_score(train_labels, prediction_values_train, average="weighted", labels=[0,1,2,3])
        
        confusion_matrix_class = confusion_matrix(test_labels, prediction_values,labels=[0,1,2,3])
        precision_classSK = precision_score(test_labels, prediction_values, average="weighted", labels=[0,1,2,3])
        recall_classSK = recall_score(test_labels, prediction_values, average="weighted", labels=[0,1,2,3])
    
    metrics = {
            "train_accuracy": round(acc_final_train,2),
            "test_accuracy": round(acc_final_test,2),
            "confusion_matrix": confusion_matrix_class,
            "precision_train":round(precision_train,2),
            "precision_test": round(precision_classSK,2),
            "recall_train": round(recall_train,2),
            "recall_test": round(recall_classSK,2),
            "execution_dir": logdir,
            "execution_time": end - start
		}
    write_metrics_to_file(metrics)
    return metrics