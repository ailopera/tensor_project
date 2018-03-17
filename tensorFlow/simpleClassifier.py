# Ejemplo de clasificador utilizando la api de alto nivel de tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score, log_loss
from datetime import datetime
# Configuramos logs para que los lea tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "logs"
tag = "simpleClassifier"
logdir = "{}/run-{}-{}/".format(root_logdir,tag, now)


mnist = input_data.read_data_sets("/tmp/data")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

config = tf.contrib.learn.RunConfig(tf_random_seed=42)

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

# Creamos un clasificador con dos capas ocultas (una con 300 y la otra con 100 neuronas)
#  Y una capa de salida softmax con 10 neuronas (una por clase)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=feature_columns, config=config)
dnn_clf = tf.contrib.learn.SkCompat(dnn_clf)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

# Entrenamos el clasificador
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

# Obtenemos metricas
# y_pred = list(dnn_clf.predict(X_test))
y_pred = dnn_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred['classes'])
log_loss = log_loss(y_test, y_pred_proba)

print("> Accuracy: ", accuracy)
print("> Log loss: ", log_loss)
