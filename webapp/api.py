from flask import Flask
from flask import request, jsonify
# Ejecucion 
# $ export FLASK_APP=hello.py
# $ flask run --hosr=0.0.0.0
import core
import gensim
from gensim.models import Word2Vec, KeyedVectors

app = Flask('NewsClassifier')

print(">> Loading word2vec Model...")
word2vec_model = '../../GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    

@app.route('/predictStance', methods=['POST'])
def predictStance():
    if 'body' in request.form and 'headline' in request.form:
        body = request.form['body']
        headline = request.form['headline']
        stance = core.predictStance(headline, body, model)
        return jsonify({'stance': stance})
