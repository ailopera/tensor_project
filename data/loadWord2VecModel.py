import gensim
from gensim.models import Word2Vec
import sys

# Obtiene los términos más cercanos y los imprime
def get_most_similar( word):
    similars = model.most_similar(word)
    for word in similars:
        print("> ",word[0],": ", word[1]) 

inputPath = sys.argv[1] 
binModel = len(sys.argv) == 3 and sys.argv[2] == False
print("BinModel: ", binModel, len(sys.argv), sys.argv[2])
print(">>> DOESNT_MATCH: Obtiene el término disonante con respecto al resto")
if not binModel:
    #model = gensim.models.Word2Vec.load("300features_15minwords_10contextALL")
    model = gensim.models.Word2Vec.load(inputPath)
else:
    model = gensim.models.KeyedVectors.load_word2vec_format(inputPath, binary=True)

# Paso 0: Representación numerica de los datos
# The word2Vec model trained consists on a feature vector for each word in the vocabulary,
# stored in a numpy array called "syn0"
# print("Tipo de la estructura de word embeddings: Type(model.wv.syn0)", type(model.wv.syn0))
print("Model shape (Tamaño del vocabulario, tamaño del vector de features): ", model.wv.syn0.shape)
try:
    # Paso 1: Exploración inicial del modelo
    # The doesht_math function will try to deduce which word in a set is most dissimilar form the others
    print(">>> doesnt_match: Obtiene el término disonante con respecto al resto")
    #print("---------------------------------------------------------------------------------------------")
    print("> Man, woman, child, kitchen: ", model.doesnt_match("man woman child kitchen".split()))
    print("> Dog, snake, horse, glass: ", model.doesnt_match("dog snake horse glass".split()))

    print("---------------------------------------------------------------------------------------------")
    print("> france, england, germany, berlin: ", model.doesnt_match("france england germany berlin".split()))
    print("> Paris, berlin, london, austria: ",model.doesnt_match("paris berlin london austria".split()))

    print("---------------------------------------------------------------------------------------------")
    print("> Streetlight, car, semaphore, mountain: ",model.doesnt_match("streetlight car semaphore mountain".split()))
    print("> Forest, team, queue, person: ",model.doesnt_match("forest team queue person".split()))
    # print("#############################################################################################")
    print(" ")

    print(">>> most_similar_cosmul: Obtiene el término más cercano tras aplicar una operación de sustracción sobre los vectores")
    forest_similar = model.wv.most_similar_cosmul(positive=['forest', 'person'], negative=['tree'])
    molecule_similar = model.wv.most_similar_cosmul(positive=['molecule'], negative=['group'])
    polymer_similar = model.wv.most_similar_cosmul(positive=['polymer'], negative=['natural'])
    print("embedded[forest] + embedded[person] - embedded[group]: ", forest_similar)
    print("embedded[molecule] - embedded[group]: ", molecule_similar)
    print("embedded[polymer] - embedded[natural]: ", polymer_similar)
    # We can also use the most_similar function to get insight into the model's wordclusters
    print(">>> MOST SIMILAR QUESTIONS")
    # print("> Man: ",get_most_similar("man"))
    # print("--------------------------------------------------------------------------------------------------------")
    print("> Woman: ")
    get_most_similar("woman")
    print("--------------------------------------------------------------------------------------------------------")
    print("> Queen: ")
    get_most_similar("queen")
    print("--------------------------------------------------------------------------------------------------------")
    print("> Obama: ")
    get_most_similar("obama")
    #print("---------------------------------------------------------------------------------------------")
    #print("> Trump: ", model.most_similar("trump")) # Not in vocabulary

    print("--------------------------------------------------------------------------------------------------------")
    #print("> Salary: ", model.most_similar("salary")) # Not in vocabulary
    print("> War: ")
    get_most_similar("war")
    print("--------------------------------------------------------------------------------------------------------")
    print("> Crisis: ")
    get_most_similar("crisis")
    print("--------------------------------------------------------------------------------------------------------")
    # model.most_similar("horrible") #Not in vocabulary
    # model.most_similar("universal") # Not in vocabulary
    print("> Global: ")
    get_most_similar("global")
    print("---------------------------------------------------------------------------------------------------------")
    print("> Love: ")
    get_most_similar("love")
    print("---------------------------------------------------------------------------------------------------------")
    print("> Money: ")
    get_most_similar("money")
    print("---------------------------------------------------------------------------------------------------------")

    #print("----------------------------------------")
    get_most_similar("money")
    print("---------------------------------------------------------------------------------------------------------")
except KeyError:
    print("El término no se encuentra en el modelo")
