

### UTILIZANDO la red de neuronas creada ###
# Ahora que tenemos la red de neuronas entrenada, podemos utilizarla para hacer predicciones
# Para ello reutilizamos la fase de creacion, y la fase de ejecucion seria la siguiente

with tf.Session() as sess:
    server.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = mnist.test.images[:20] # Las imagenes a predecir (escaladas de 0 a 1)
    Z = logits.eval(feed_dict={X: X_new_scaled})
    # Si quisieramos saber las probabilidades para cada clase, seria necesario aplicar la funcion de softmax
    # Pero si solo queremos obtener la clase, simplemente cogemos la que tiene mayor probabilidad, con argmax
    y_pred = np.argmax(Z, axis=1)

    print("Predicted classes: ", y_pred)
    print("Actual classes: ", mnist.test.labels[:20])
