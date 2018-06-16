from flask import Flask
from flask import request, jsonify, render_template
from flask_restful import Resource, Api 

from json import dumps 

import core
import gensim
from gensim.models import Word2Vec, KeyedVectors

# Ejecucion -- Modo desarrollo
# $ export FLASK_APP=api.py
# $ flask run --host=0.0.0.0


app = Flask('NewsClassifier')
api = Api(app)

print(">> Loading word2vec Model...")
word2vec_model = '../../GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
print(">> Done! ")

# Manejadores de la api
class Stances(Resource):
    def post(self):
        if 'body' in request.form and 'headline' in request.form:
            body = request.form['body']
            headline = request.form['headline']
            stance = core.predictStance(headline, body, model)
            return jsonify({'stance': stance})
        else:
            return jsonify({'error': 'You must provide both body and articleBody params'})

# Recursos de la API
api.add_resource(Stances,'/stances')

if __name__ == '__main__':
    app.run(port='5000')
