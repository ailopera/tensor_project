import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Ejemplo de implementacion de gradient descent
housing = fetch_california_housing()
m,n = housing.data.shape

housing_data_plus_bias= np.c_[np.ones((m,1)), housing.data]

n_epochs = 1000
learning_rate = 0.1


X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="x")
Y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# El gradiente descendiente requiere reescalar los vectores de features.
# El escalado se puede hacer tanto con scikit learn como con tensorflow
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

# gradients = 2/m * tf.matmul(tf.transpose(X),error)
# Otra opcion para calcular el gradiente es utilizar autodiff, que commputa el gradiente automaticamente
# La funcion del gradiente toma como parametro un operando (en nuestro caso mse) y una lista de variables (en nuestro caso theta)
# y crea una lista de operandos (uno por variable) para computar el gradiente del operando con respecto a cada varaible
gradients = tf.gradients(mse,[theta])

# Con assign creamos un nodo que asignara un nuevo valor a la variable
# En nuestro caso este nodo implementa el paso del batch gradient descent
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Ejecutamos n_epoch iteraciones de entrenamiento
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch ", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
