from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_sample(X, y)
print(sorted(Counter(y_resampled).items()))
