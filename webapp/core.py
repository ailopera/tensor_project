import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '../data/')
import vectorAverage 
import word2VecModel

#BASE_PATH = "../data/models/"
BASE_PATH = "../data/models/model/"
MODEL_NAME = BASE_PATH + "my_model_final.ckpt"
class_names= ["agree", "disagree", "discuss", "unrelated"]

NUM_FEATURES = 300

"""
    Obtiene el vector de caracteristicas de un texto dado
"""
def generateFeatureVector(text,model):
    cleaned_text = word2VecModel.news_to_wordlist(text, remove_stopwords=True, clean_text=True)
    featureVector = vectorAverage.getAvgFeatureVecs([cleaned_text], model, NUM_FEATURES)
    return featureVector

"""
Obtiene la prediccion a partir del vector de titular y cuerpo de noticia
"""
def predictStance(headline, articleBody, model):
    print(">> Obteniendo vectores de embedding de la entrada")
    headline_embedding = generateFeatureVector(headline, model)
    articleBody_embedding = generateFeatureVector(articleBody, model)
    input_sample = np.append(headline_embedding, articleBody_embedding)
    print(">> Shape: ", input_sample.shape)
    tf.reset_default_graph()
    #n_inputs = input_sample.shape[0]
    #X = tf.placeholder(tf.float32, shape=(None), name="X")
    #y = tf.placeholder(tf.int64, shape=(None), name="y")
    #keep_prob = tf.placeholder(tf.float32, shape=(None), name="keep_prob")

    
    default_label = 0
    saver = tf.train.import_meta_graph(MODEL_NAME + ".meta")  # this loads the graph structure
    # prediction = tf.get_default_graph().get_tensor_by_name("prediction:0") # not shown in the book

    with tf.Session() as sess:
        #saver.restore(sess, MODEL_NAME)  # this restores the graph's state
        #print("tf.graphKeys: ", [n.name for n in tf.get_default_graph().as_graph_def().node])
        new_saver = tf.train.import_meta_graph(BASE_PATH + 'my_model_final.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(BASE_PATH))
        
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        y = graph.get_tensor_by_name("y:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        stance = sess.run('prediction:0', feed_dict={X: [input_sample], y: default_label, keep_prob: 1.0})
        print(">> Prediction: ", stance)
        return class_names[stance[0]]
