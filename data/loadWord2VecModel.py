import gensim
from gensim.models import Word2Vec
import sys
inputPath = sys.argv[1] 
#model = gensim.models.Word2Vec.load("300features_15minwords_10contextALL")
model = gensim.models.Word2Vec.load(inputPath)
#model = gensim.models.KeyedVectors.load_word2vec_format(inputPath, binary=True)

# Paso 0: Representación numerica de los datos
# The word2Vec model trained consists on a feature vector for each word in the vocabulary,
# stored in a numpy array called "syn0"
# print("Tipo de la estructura de word embeddings: Type(model.wv.syn0)", type(model.wv.syn0))
print("Model shape (Tamaño del vocabulario, tamaño del vector de features): ", model.wv.syn0.shape)

# Paso 1: Exploración inicial del modelo
# The doesht_math function will try to deduce which word in a set is most dissimilar form the others
print(">>> DOESNT_MATCH: Obtiene el término disonante con respecto al resto")
#print("---------------------------------------------------------------------------------------------")
print("> Man, woman, child, kitchen: ", model.doesnt_match("man woman child kitchen".split()))
#print("---------------------------------------------------------------------------------------------")
print("> france, england, germany, berlin: ", model.doesnt_match("france england germany berlin".split()))
#print("---------------------------------------------------------------------------------------------")
print("> Paris, berlin, london, austria: ",model.doesnt_match("paris berlin london austria".split()))

# print("#############################################################################################")
print(" ")
# We can also use the most_similar function to get insight into the model's wordclusters
print(">>> MOST SIMILAR QUESTIONS")
# print("> Man: ",get_most_similar("man"))
# print("--------------------------------------------------------------------------------------------------------")
print("> Woman: ",get_most_similar("woman"))
print("--------------------------------------------------------------------------------------------------------")
print("> Queen: ", get_most_similar("queen"))

print("--------------------------------------------------------------------------------------------------------")
print("> Obama: ", get_most_similar("obama"))
#print("---------------------------------------------------------------------------------------------")
#print("> Trump: ", model.most_similar("trump")) # Not in vocabulary

print("--------------------------------------------------------------------------------------------------------")
#print("> Salary: ", model.most_similar("salary")) # Not in vocabulary
print("> War: ", get_most_similar("war"))
print("--------------------------------------------------------------------------------------------------------")
print("> Crisis: ", get_most_similar("crisis"))
print("--------------------------------------------------------------------------------------------------------")
# model.most_similar("horrible") #Not in vocabulary
# model.most_similar("universal") # Not in vocabulary
print("> Global: ", get_most_similar("global"))
print("---------------------------------------------------------------------------------------------------------")
print("> Love: ", get_most_similar("love"))
print("---------------------------------------------------------------------------------------------------------")
print("> Money: ", get_most_similar("money"))
print("---------------------------------------------------------------------------------------------------------")

#print("----------------------------------------")
#print("Sample word model: ", model["love"])


# Obtiene los términos más cercanos y los imprime
def get_most_similar(model, word):
    similars = model.most_similar(word)
    for word in similar:
        print("> ",word[0],": ", word[1]) 