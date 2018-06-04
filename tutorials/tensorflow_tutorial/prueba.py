# Fichero con el ejemplo descrito en Getting Started with TensorFlow
import tensorflow as tf

# Constants takes no imputs and outputs a value it stores internally
node1 = tf.constant(3.0,tf.float32) 
node2 = tf.constant(4.0) # also tf.float32 implicitly
#print(node1,node2) 

sess = tf.Session()
#print(sess.run([node1,node2]))

node3 = tf.add(node1, node2)
#print("node3: ", node3)
#print("sess.run(node3): ", sess.run(node3))


# Placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for tf.add(a, b)
#The precedind three lines are a bit like a function or a lambda in which we define two input parameters and then an operation on them. 
#We can evaluate this graph with multiple inputs by using the feed_dict parameter to specify Tensors that provide concrete values to these placeholders

#print(sess.run(adder_node, {a:3, b:4.5}))
#print(sess.run(adder_node, {a: [1,3], b: [2,4]})) 

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:3, b:4.5}))
