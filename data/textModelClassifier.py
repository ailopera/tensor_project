# Ejemplo de clasificador utilizando tensorflow de bajo nivel
# En este caso utilizamos la funcion dense() en vez de crear una funcion propia
import tensorflow as tf
import numpy as np
from datetime import datetime
import random

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

    # print("> Len(minibatch_data): ", len(minibatch_data))
    # print("> Len(minibatch_targets): ", len(minibatch_targets))
    # print("> Data sample: ", minibatch_data[0])
    # print("> Target sample: ", minibatch_targets[0])
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
def modelClassifier(input_features, target, test_features, test_targets):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "logs"
    tag = "tensorClassifierWithDense"
    logdir = "{}/run-{}-{}/".format(root_logdir,tag, now)

    # Convertimos a enteros las clases
    train_labels = convert_to_int_classes(target)
    test_labels = convert_to_int_classes(test_targets)
    
    # Pequeño logging para comprobar que se estan generando bien 
    # for i in range(20):
    #     print(">> String label: ", target[i])
    #     print(">> Int label: ", train_labels[i])

    ### Definicion de la red ###
    train_samples = input_features.shape[0] # Numero de ejemplos

    # Hiperparametros del modelo
    n_inputs = input_features.shape[1] #Tamaño de la entrada
    n_hidden1 = 300 # Numero de neuronas de la primera capa oculta
    n_hidden2 = 100 # Numero de neuronas de la segunda capa oculta
    n_outputs = 4 # Numero de salidas/clases a predecir

    print("> Shape de los datos de entrada (entrenamiento): ", input_features.shape)
    print("> Shape de los datos de entrada (test): ", test_features.shape)
    print("> Numero de neuronas de la capa de entrada: ", n_inputs)
    print("> Numero de instancias de entrenamiento: ", train_samples)

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
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
    with tf.name_scope("eval"):
        prediction = tf.nn.in_top_k(logits, y , 1)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        recall = tf.metrics.recall(y, prediction)
        precision = tf.metrics.precision(y, prediction)
        confusion_matrix = tf.confusion_matrix(y, prediction)
        mean_accuracy = tf.reduce_mean(accuracy)
        
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Imprimimos el grafo para verlo desde tensorflow
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    #### Fase de ejecucion ###
    # Cargamos el dataset
    # mnist = input_data.read_data_sets("/tmp/data/")
    n_epochs = 20
    batch_size = 50

    n_iterations = round(train_samples / batch_size)

    # Entrenamos el modelo. Usamos minibatch gradient descent 
    # (en cada iteracion aplicamos el gradiente descendiente sobre una submuestra aleatoria de los datos de entrenamiento)
    #  Al final de cada epoch computamos el accuracy sobre uno de los batches.
    with tf.Session() as sess:
        # Inicializamos las variables globales del grafo
        init.run()
        # Realizamos el entrenamiento fijando en n_epochs
        for epoch in range(n_epochs):
            for iteration in range(n_iterations):
                # X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch, y_batch = next_batch(batch_size, input_features, train_labels)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
            # Obtenemos el accuracy de los datos de entrenamiento y los de tests    
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_features, y: test_labels})
            print(epoch, "Train accuracy: ", acc_train, " Test accuracy: ", acc_test)

            # Sacamos el valor actual de los dos accuracy en los logs para visualizarlo en tensorboard
            acc_train_summary = tf.summary.scalar('Train Accuracy', acc_train)
            acc_test_summary = tf.summary.scalar('Test Accuracy', acc_test)
        
        # Ejecutamos las metricas finales
        sess.run(tf.local_variables_initializer())
        #acc_train = sess.run(mean_accu, )
        #acc_train = sess.run(mean_accu, )
        recall_class = sess.run(recall, feed_dict={X: test_features, y: test_labels})
        precision_class = sess.run(precision, feed_dict={X: test_features, y: test_labels})
        confusion_matrix_class = sess.run(confusion_matrix, feed_dict={X: test_features, y: test_labels})
        # Guardamos la version actual del modelo entrenado
        save_path = saver.save(sess, "./my_model_final.ckpt")
        
        metrics = {
	 	"train_accuracy": acc_train,
		"test_accuracy": acc_test,
		"confusion_matrix": confusion_matrix_class,
		"average_precision": precision_class[1],
		"recall": recall_class[1]
		}
        return metrics
