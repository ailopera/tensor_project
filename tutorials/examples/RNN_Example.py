import tensorflow as tf


#  UTILIZANDO BASIC CELL
# n_steps = 2
# n_inputs = 3
# n_neurons = 5
# # Representa todas las entradas del modelo
# # Es un vecotr mini_batch_size x n_steps x n_inputs
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# # Sacamos un elemento de la entrada despues de permutar la entrada las dos primeras dimensiones
# x_seqs = tf.unstack(tf.transpose(X, pem=[1,0,2]))

# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, x_seqs, dtype=tf.float32)

# # Lo metemos en el array de resultados, reaordenando las dimensiones
# outputs = tf.transpose((tf.stack(output_seqs)), perm=[1,0,2])

# init = tf.global_variables_initializer()


# X_batch  =  np.array([
#         # t = 0      t = 1 
#         [[0, 1, 2], [9, 8, 7]], # instance 1
#         [[3, 4, 5], [0, 0, 0]], # instance 2
#         [[6, 7, 8], [6, 5, 4]], # instance 3
#         [[9, 0, 1], [3, 2, 1]], # instance 4
#     ])

# with tf.Session() as sess:
#     init.run()
#     outputs_val = outputs.eval(feed_dict={X: X_batch})


# USANDO DYNAMIC RNN
n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

init = tf.global_variables_initializer()
X_batch  =  np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])


with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})


