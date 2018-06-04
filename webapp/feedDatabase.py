    # Script que rellena la base de datos para la aplicacion
import pymongo
from pymongo import MongoClient
import pandas as pd

LABEL = 'competition'

base_dir = "../data/fnc-1-original/"
stances_file = base_dir + LABEL + "_test_stances.csv"
bodies_file = base_dir + LABEL + "_test_bodies.csv"

stancesData = pd.read_csv(stances_file, header=0, delimiter=',', quoting=1)
bodiesData = pd.read_csv(bodies_file, header=0, delimiter=',', quoting=1)

client = MongoClient('localhost', 27017)
db = client['newsData']
# Abrimos el fichero de bodies original y lo almacenamos en la BBDD
print(">> Writting stances data...")
print(">> Total Stances: ", stancesData.shape)
for line in stancesData.iterrows(): 
    stance = {
        "headline": line["Headline"],
        "bodyId": line["Body ID"],
        "correctStance": line["Stance"],
        "partition": LABEL
    }
    db.Stances.insert(stance)

# Abrimos el fichero de headline+stances original y lo almacenamos en la BBDD
print(">> Writing bodies data...")
print(">> Total Bodies: ", bodiesData.shape)
for line in bodiesData.iterrows():
    body = {
        "bodyId": line["Body ID"],
        "articleBody": line["articleBody"],
        "partition": LABEL
    }
    db.Bodies.insert(body)
