import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, '../data/')
import vectorAverage 
import word2VecModel

BASE_PATH = "../data/models/"
MODEL_NAME = BASE_PATH + "my_model_final.ckpt"
class_names= ["agree", "disagree", "discuss", "unrelated"]

NUM_FEATURES = 300

"""
    Obtiene el vector de caracteristicas de un texto dado
"""
def generateFeatureVector(text,model):
    cleaned_text = word2VecModel.news_to_wordlist(text, remove_stopwords=True, clean_text=True)
    featureVector = vectorAverage.getAvgFeatureVecs(cleaned_text, model, NUM_FEATURES)
    return featureVector

"""
Obtiene la prediccion a partir del vector de titular y cuerpo de noticia
"""
def predictStance(headline, articleBody, model):
    tf.reset_default_graph()

    headline_embedding = generateFeatureVector(headline, model)
    articleBody_embedding = generateFeatureVector(articleBody, model)
    input_sample = np.append(headline_embedding, articleBody_embedding)
    
    default_label = 0
    saver = tf.train.import_meta_graph(MODEL_NAME + ".meta")  # this loads the graph structure
    # prediction = tf.get_default_graph().get_tensor_by_name("prediction:0") # not shown in the book

    with tf.Session() as sess:
        saver.restore(sess, MODEL_NAME)  # this restores the graph's state
        stance = sess.run('prediction:0', feed_dict={X: input_sample, y: default_label, keep_prob: 1.0}) # not shown in the book
        print(">> Prediction: ", stance)
        return class_names[stance]