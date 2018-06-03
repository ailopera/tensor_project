from cleanData import cleanTextData

# Script que limpia los datasets para stance detection


# Body -> textTag: "articleBody"
#          outputFilePath:"./fnc-1-original/cleanDatasets/train_bodies_clean.csv"
# Primera parte del trabajo: experimentacion con vectoraverage y bow + MLP

### STANCES ###
# inputStancesPath = "./fnc-1-original/train_stances.csv"
# outputStancesPath = "./fnc-1-original/cleanDatasets/train_stances_clean.csv"
# print(">>> Cleaning out Stances Data")
# cleanTextData(True,inputStancesPath, outputStancesPath, False)


# # ### BODIES ###
# inputBodiesPath = "./fnc-1-original/train_bodies.csv"
# outputBodiesPath = "./fnc-1-original/cleanDatasets/train_bodies_clean.csv"
# print(">>> Cleaning out Body Data")
# cleanTextData(False,inputBodiesPath, outputBodiesPath, False)


# Segunda parte del trabajo: experimentacion con flujos de texto + RNN
inputStancesPath = "./fnc-1-original/train_stances.csv"
outputStancesPath = "./fnc-1-original/cleanDatasets/RNN/train_stances_clean.csv"
print(">>> Cleaning out Stances Data")
cleanTextData(True,inputStancesPath, outputStancesPath, False, maintainDots = True)


# ### BODIES ###
inputBodiesPath = "./fnc-1-original/train_bodies.csv"
outputBodiesPath = "./fnc-1-original/cleanDatasets/RNN/train_bodies_clean.csv"
print(">>> Cleaning out Body Data")
cleanTextData(False,inputBodiesPath, outputBodiesPath, False, maintainDots = True)