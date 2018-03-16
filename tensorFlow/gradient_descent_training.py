import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Ejemplo de implementacion de gradient descent
housing = fetch_california_housing()
m,n = housing.data.shape

housing_data_plus_bias=np.c_[np.ones((m,1)), housing.data]
# El gradiente descendiente requiere reescalar los vectores de features.
# El escalado se puede hacer tanto con scikit learn como con tensorflow
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]


n_epochs = 1000
learning_rate = 0.1

batch_size = 100
n_batches = int(np.ceil(m/batch_size))

# Para el entrenamiento del modelo con los datos usaremos X e Y como placeholders en vez de como constantes
# En este programa vamos a implementar un Mini-batch Gradient Descent
# Si en una de las dimensiones ponemos None, estamos queriendo decir que puede ser de cualquier tamaño
X = tf.placeholder(tf.float32, shape=(None, n+1), name="x")
y = tf.placeholder(tf.float32, shape=(None, n+1))
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Otra de las opciones para calcular gradientes es la de utilizar un optimizador
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Se pueden utilizar otros optimizadores como momentum, que converge mucho mas rapido que el gradiente descendiente
# optimizer = tf.train.momentumOptimizer(learning_rate(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


init = tf.global_variables_initializer()

# En la fase de ejecución, obtenemos los mini-batches uno a uno, proveemos le valor de X e Y
# con el parametro de feed_dict, àra cuando evaluamos un nodo que depende de otro
def fetch_batch(epoch,batch_index, batch_size):
    # TODO: Cargamos los datos del disco
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    # Ejecutamos n_epoch iteraciones de entrenamiento
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        best_theta = theta.eval()