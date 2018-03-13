import tensorflow as tf

# Creamos dos variables globales
x = tf.Variable(3, name="x")
y = tf.Variable(2, name="y")

# F es la funcion que define el grafo
f = x*x*y + y + 2

# sess = tf.Session()
# sess.run(x.initialilizer)
# sess.run(y.initialilizer)
# result = sess.run(f)
# print("> Resultado: ", result)
# sess.close()

# Otra opcion, para no tener que llamar a sess.run todo el rato es hacer lo siguiente:
# La sesion es seteada como la sesion por defecto, y se cierra automaticamente al final del bloque
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
#     print("> Resultado: ", result)



# Otra opcion para inicializar las variables es la de crear un nodo en el grafo que inicializara todas las variables cuando se ejecute
init = tf.global_variables_initializer() # preparamos un nodo init

with tf.Session() as sess:
    init.run() # Inicializa todas las variables
    result = f.eval()
    print("> Resultado: ", result)

