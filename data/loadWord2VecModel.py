import gensim
from gensim.models import Word2Vec
model = gensim.models.Word2Vec.load("300features_40minwords_10contextBODIES")

# Paso 1: Exploración inicial del modelo
# The doesht_math function will try to deduce which word in a set is most dissimilar form the others
print(">>> DOESNT MATCH QUESTIONS")
print("> Man, woman, child, kitchen: ", model.doesnt_match("man woman child kitchen".split()))

print("> france, england, germany, berlin: ", model.doesnt_match("france england germany berlin".split()))
print("> Paris, berlin, london, austria: ",model.doesnt_match("paris berlin london austria".split()))

# We can also use the most_similar function to get insight into the model's wordclusters
print(">>> MOST SIMILAR QUESTIONS")
print(">Man: ",model.most_similar("man"))
print("> Queen: ", model.most_similar("queen"))
print("> Obama: ", model.most_similar("obama"))

# model.most_similar("horrible") #Not in vocabulary
# model.most_similar("universal") # Not in vocabulary
print("> Global: ", model.most_similar("global"))
print("> Love: ", model.most_similar("love"))
print("> Money: ", model.most_similar("money"))


# Paso 2: Representación numerica de los datos
# The word2Vec model trained consists on a feature vector for each word in the vocabulary,
# stored in a numpy array called "syn0"
print("Type model.syn0", type(model.wv.syn0))
print("Model shape (#words in the model vocabulary, size of the feature vector): ", model.wv.syn0.shape)

print("Sample word model: ", model["love"])
