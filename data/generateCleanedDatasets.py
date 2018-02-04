from cleanData import cleanTextData

# Script que limpia los datasets para stance detection


# Body -> textTag: "articleBody"
#          outputFilePath:"./fnc-1-original/cleanDatasets/train_bodies_clean.csv"
### STANCES ###
inputStancesPath = "./fnc-1-original/train_stances.csv"
outputStancesPath = "./fnc-1-original/cleanDatasets/train_stances_clean.csv"
print(">>> Cleaning out Stances Data")
cleanTextData(True,inputStancesPath, outputStancesPath, True)


# ### BODIES ###
# inputBodiesPath = "./fnc-1-original/train_bodies.csv"
# outputBodiesPath = "./fnc-1-original/cleanDatasets/train_bodies_clean.csv"
# print(">>> Cleaning out Body Data")
# cleanTextData(False,inputBodiesPath, outputBodiesPath, True)