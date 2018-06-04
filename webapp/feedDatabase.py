    # Script que rellena la base de datos para la aplicacion
import pymongo
from pymongo import MongoClient

LABEL = 'competition'

base_dir = "../data/fnc-1-original/"
stances_file = base_dir + LABEL + "_test_stances.csv"
bodies_file = base_dir + LABEL + "_test_bodies.csv"


client = MongoClient('localhost', 27017)
db = client.newsData()
# Abrimos el fichero de bodies original y lo almacenamos en la BBDD
with open(stances_file, 'r') as stances:
    for line in stances: 
        stance = {
            "headline": line["Headline"],
            "bodyId": line["Body ID"],
            "correctStance": line["Stance"],
            "label": LABEL
        }
        db.Stances.insert(stance)

# Abrimos el fichero de headline+stances original y lo almacenamos en la BBDD
with open(bodies_file, 'r') as bodies:
    for line in bodies:
        body = {
            "bodyId": line["Body ID"],
            "articleBody": line["articleBody"],
            "label": LABEL
        }
        db.Bodies.insert(body)
