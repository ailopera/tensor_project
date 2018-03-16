import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from datetime import datetime

now = datetime.utcnow().strftime(%Y%m%d%H%M%S)
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Ejemplo de implementacion de gradient descent
### FASE de Creacion ###
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
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Otra de las opciones para calcular gradientes es la de utilizar un optimizador
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Se pueden utilizar otros optimizadores como momentum, que converge mucho mas rapido que el gradiente descendiente
# optimizer = tf.train.momentumOptimizer(learning_rate(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

# Inicializador de las variables x e y
init = tf.global_variables_initializer()

# Una vez entrenado el modelo podemos guardar sus parametros
# Tambien resulta util para guardar checkpoints del trabajo realizado por si el servidor se cae
# Para ello creamos un nodo de guarda al final de la fase de construccion, despues de crear todos los nodos
saver = tf.train.Saver()


def fetch_batch(epoch,batch_index, batch_size):
    # Cargamos los datos del disco
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


# Para poder visualizar los datos en tensorboard:
# Creamos un nodo que evalua el valor de mse y lo escribbe en un log compatible con tensorboard llamado summary
mse_summary = tf.summary.scalar('MSE', mse)
# Creamos un filewriter para poder escribir summaries en logfiles, en el directorio de logs
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

### FASE de ejecucion ###
# En la fase de ejecución, obtenemos los mini-batches uno a uno, proveemos le valor de X e Y
# con el parametro de feed_dict, para cuando evaluamos un nodo que depende de otro

# En la fase de ejecucion simplemente llamamos a la funcion save, con la referencia de la sesion y la ruta del fichero checkpoint
with tf.Session() as sess:
    sess.run(init)

    # Ejecutamos n_epoch iteraciones de entrenamiento
    for epoch in range(n_epochs):
        # if epoch % 100 == 0: #Hacemos un checkpoint cada 100 epochs
        #     print("Epoch", epoch, "MSE =", mse.eval())
        #     save_path = saver.save(sess,"./my_model.ckpt")
        # sess.run(training_op)

        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        best_theta = theta.eval()
    # save_path = saver.save(sess, "./my_model_final.ckpt")


# Para restaurar un modelo:
# with tf.Session() as sess:
#     server.restore(sess, "./my_model_final.ckpt")
file_writer.close()