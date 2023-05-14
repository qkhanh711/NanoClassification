from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

import warnings
from sklearn.exceptions import FitFailedWarning, DataConversionWarning

from utils import *
from extract_feature_AE import *
from extract_feature_PCA import *

def test_models(X_train, y_train, X_test, y_test):
    models = [LogisticRegression(C = 0.1, penalty = 'l2', solver = 'liblinear'),
            SVC(C = 1.0, degree = 2, gamma = 'scale', shrinking = True,  probability=True, kernel = 'linear'),
            KNeighborsClassifier(algorithm = 'auto',p = 1, weights = 'distance', n_neighbors = 3, metric = 'cosine'),
            DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, min_samples_leaf= 1, min_samples_split = 5, splitter= 'random',  random_state = 42),
            RandomForestClassifier(criterion = 'gini', max_depth = 6, min_samples_leaf = 2, min_samples_split = 5, n_estimators = 100,  random_state = 42),
            BernoulliNB(alpha=0.1)]

    for i in range(len(models)):
        print(model_predict(X_train, y_train, X_test, y_test, models[0])['Predict'])
        print(y_test.T)