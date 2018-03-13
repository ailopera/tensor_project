import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(2, name="y")

f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
