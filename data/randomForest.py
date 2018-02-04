print("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier

# The random forest algorithm is included in scikit-learn. Random Forest uses many tree-based classifiers to
# make predictions, hence the "forest"
# Below, we set the number of trees to 100 as a reasonable default value. More trees may (or may not) perform
# better, but will certainly take longer to run

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response
# variable

# This may take a few minutes to run
forest = forest.fit(train_data_features, train["stance"])