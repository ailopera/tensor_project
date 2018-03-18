# Ejemplo de clasificador utilizando tensorflow de bajo nivel
# En este caso utilizamos la funcion dense() en vez de crear una funcion propia
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
tag = "tensorClassifierWithDense"
logdir = "{}/run-{}-{}/".format(root_logdir,tag, now)

n_inputs = 28*28 #MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

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
    correct = tf.nn.in_top_k(logits, y , 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Imprimimos el grafo para verlo desde tensorflow
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#### Fase de ejecucion ###
n_epochs = 20
n_batches = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy: ", acc_train, " Test accuracy: ", acc_test)
    
    save_path = saver.save(sess, "./my_model_final.ckpt")
    
