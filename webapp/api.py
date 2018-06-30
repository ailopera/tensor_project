from flask import Flask
from flask import request, jsonify, render_template
from flask_restful import Resource, Api 

from json import dumps 
#from data import model
import core
import gensim
from gensim.models import Word2Vec, KeyedVectors


from flask_cors import CORS

# Ejecucion -- Modo desarrollo
# $ export FLASK_APP=api.py
# $ flask run --host=0.0.0.0

model = None
app = Flask('NewsClassifier')
api = Api(app)
cors = CORS(app, resources={r"/stances/*": {"origins": "*"}})

word2vec_model = '../../GoogleNews-vectors-negative300.bin'
print(">> Loading word2vec Model...")
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
print(">> Done! ")


# Manejadores de la api
class Stances(Resource):
    
    def post(self):        
        if 'body' in request.form and 'headline' in request.form:
            body = request.form['body']
            headline = request.form['headline']
            full_predictions, stance = core.predictStance(headline, body, model)
            return jsonify({
                'headline': headline,
                'body': body,
                'stance': stance,
                'predictions': full_predictions})
        else:
            return jsonify({'error': 'You must provide both body and headline params'})

# Recursos de la API
api.add_resource(Stances,'/stances')

if __name__ == '__main__':
    app.run(port='5000')
