# Ejemplo de clasificador utilizando tensorflow de bajo nivel
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
tag = "tensorClassifier"
logdir = "{}/run-{}-{}/".format(root_logdir,tag, now)

# We will implement MiniBatch Gradient Descent to train it on the MNIST Dataset
#  The first step is the construction phase, building the TensorFlow graph
# The second step is the execution phase, where you actually run tthe graph to train the model

#  NOTA: El dataset MNIST contiene imagenes de digitos escritos a mano

### Construction Phase ###
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# Utilizamos nodos placeholder para representar los datos de entrenamiento y los targets
#  The input will be a 2D tensor (matrix), with instances along the first dimension
#  and features along the second dimension, and we know that the number of features is going to be 
# 28 x 28 (one feature per pixel). but we don't know yet how many instances each trainin batch will contain.

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# X actuara como la capa de entrada del modelo, y durante la fase de ejecucion sera reemplazado
# por un batch de entrenamient cda vez

# Creamos la red de neuronas con dos capas ocultas y la capa de salida
# Las capas ocultas son practicamente iguales, solo se diferencian por las entradas a las que se encueentran conectadas
# y por el numero de neuronas que tienen

# Utilizamos como funcion de activacion softmax
# Creamos una funcion que define una capa de neuronas
def neuron_layer(X, n_neurons, name, activation=None):
    # 1. Creamos un name scope con el nombre de la capa, que contendra todos los nodos de computacion para esta capa de neuronas
    with tf.name_scope(name):
        # 2. Obtenemos el numero de entradas a partir del shape de X. 
        # La primera dimension indica el numero de instancias, la segunda el numero de entradas
        n_inputs = int(X.get_shape()[1])
        # 3. Creamos una variable W que contendra la matriz de pesos para cada conexion entrada-neurona
        # La inicializamos de forma aleatoria, con una distribucion truncada normal Gaussiana, con la desviacion estandar proporcionada
        # Esta desviacion hace que el algoritmo converga mucho mas rapido
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weigths")
        # 4. Creamos una variable de bias, un parametro de bias por neurona.
        # En este caso lo seteamos a 0
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        # 5. Creamos un subgrafo para computar la salida
        z = tf.matmul(X,W) + b
        if activation=="relu":
            return tf.nn.relu(z)
        else:
            return z

# Creamos la red neuronal. La primera capa oculta recibe como entrada X. La segunda toma como entrada  la primera capa oculta
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X,n_hidden1,"hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1,n_hidden2, "hidden2", activation="relu")
    # Logits es la salida de la capa antes de computar la funcion softmax
    logits = neuron_layer(hidden2, n_outputs, "outputs")


# Una vez definida la red neuronal, necesitamos definir la funcion de coste que usaremos para entrenar el modelos
# Para ello utilizaremos cross_entropy.
# Esta funcion penalizara los modelos que estimen una baja probabilidad para la clase objetivo
# TensorFlow proporciona varias funciones para computar la entropia cruzada. Utilizamos la siguiente
with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Definimos un GradientDescentOptimizer que realizara un ajuste de los parametros del modelo para minimizar la funcion de coste
learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Finalmente evaluamos la performance del modelo con accuracy
# Para cada instancia, determinamos si la preciccion de la red neuronal ess cirrecta,
# Para ello utilizamos la funcion in_top_k()
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y , 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Creamos un nodo para inicializar todas las variables y un nodo de guarda para guardar nuestro modelo entrenado en disco
init = tf.global_variabless_initializer()
saver = tf.train.Saver()

# Escribimos el grafo
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#### EXECUTION PHASE ###
# Cargamos el dataset que vamos a utilizar
# TensorFlow ofrece un helper que obtiene los datos, los escala entre 0 y 1 y los agita
#  y proporciona una funcion para cargar una porcion (minibatch) cada vez
mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 40
batch_size = 50

# Abrimos una sesion de tensorflow
with tf.Session() as sess:
    # inicializamos las variables
    init.run()
    # En cada epoch, el codigo itera a traves de un numero de mmini-batches
    # que se corresponden con el conjunto de entrenamiento
    # Obtenemos cada batch con la funcion next_batch
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples): # Batch size
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        # Al final de cada epoch, el codigo evalua el modelo en el ultimo mini-batch y sobre el conjunto de entrenamiento completo
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
        print(epoch, "Train accuracy: ", acc_train, " Val accuracy: ", acc_val)
    
    # Guardamos los parametros del modelo en disco
    save_path = saver.save(sess, "./my_model_final.ckpt")

#TODO: Mostrar el grafo con tensorboard





