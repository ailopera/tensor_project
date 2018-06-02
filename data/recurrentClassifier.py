import tensorflow as tf
import numpy as np
from datetime import datetime
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import csv
import os
import time

EXECUTION_TAG = "_arquitectura_singleLayer"

### Funciones auxiliares
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
def modelClassifier(input_features, target, test_features, test_targets, configuration=None):
    tf.reset_default_graph() 
    date = datetime.utcnow().strftime("%Y%m%d")
    hour = datetime.utcnow().strftime("%H%M%S")
    start = time.time()
    # root_logdir = "testLogs"
    execution_date = time.strftime("%m-%d")
    root_logdir = "RNNLogs/" + execution_date + EXECUTION_TAG
    tag = "RNNClassifier"
    config_tag = "Prueba"
    #config_tag = hyperparams.get("config_tag" , default_hyperparams["config_tag"])
    subdir = date
    logdir = "{}/{}/run-{}-{}-{}/".format(root_logdir, subdir , tag, config_tag, hour)

    # Convertimos a enteros las clases
    train_labels = convert_to_int_classes(target)
    test_labels = convert_to_int_classes(test_targets)


    print(">> Input shape: ", input_features.shape)
    n_steps = 2
    n_inputs = 300
    
    n_neurons = 150
    n_outputs = 4

    # Redimensionamos los datos de test
    test_features = test_features.reshape((-1, n_steps, n_inputs))
    
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    with tf.name_scope("rnn"):
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
        logits = tf.layers.dense(states, n_outputs)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy)

    learning_rate = 0.001
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 20
    batch_size = 150
    
    train_samples = input_features.shape[0] # Numero de ejemplos
    
    n_iterations = round(train_samples / batch_size)
    print("> Numero de instancias de entrenamiento: ", train_samples)
    
    #Definimos los escalares que visualizaremos
    tf.summary.scalar('Accuracy', accuracy)
    tf.summary.scalar('Loss', final_loss)
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
            
            acc_train, summary_train = sess.run([accuracy, merged_summary_op], feed_dict={X: input_features, y: train_labels})
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