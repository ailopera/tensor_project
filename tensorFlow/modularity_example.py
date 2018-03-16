import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
tag = "RELu"
logdir = "{}/run-{}-{}/".format(root_logdir,tag, now)

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]),1)
        w = tf.Variable(tf.random_normal(w_shape), name="weigths")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X,w),b, name="z")
        return tf.maximum(z, 0., name="max")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
