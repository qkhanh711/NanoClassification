import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
from sklearn.exceptions import FitFailedWarning, DataConversionWarning
import time

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

import warnings
from sklearn.exceptions import FitFailedWarning, DataConversionWarning

paths = ["Data for Nano-AI/4-nitrophenol", "Data for Nano-AI/Carbaryl", "Data for Nano-AI/Chloramphenicol", "Data for Nano-AI/Congo Red", "Data for Nano-AI/Crystal Violet", 
         "Data for Nano-AI/Glyphosate", "Data for Nano-AI/Methylene Blue", "Data for Nano-AI/Thiram", "Data for Nano-AI/Tricyclazole", "Data for Nano-AI/Urea"]
labels = ["4_nitrophenol", "Carbaryl", "Chloramphenicol", "Congo red", "Crystal Violet", "Glyphosate", "Methylene", "Thiram", "Tricylazole", "Urea"]
compare = ["Tricyclazole.txt", "Tricyclazole So sanh.txt"]

file_names = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt']

def plot_data(paths, labels, file_names, nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows= nrows, ncols= ncols, figsize= figsize)
    axes = axes.flatten()

    for i, path in enumerate(paths):
        path_data = os.path.join(path, file_names)
        data = pd.read_csv(path_data, sep="\t")

        x = data.iloc[-1000:, 0].values
        y = data.iloc[-1000:, 1].values
        ax = axes[i]
        ax.set_xlabel("Raman Shift")
        ax.set_ylabel("a.u.")
        ax.plot(x, y)
    
        ax.set_title(f"{labels[i]}")
    
    plt.tight_layout()  
    plt.show()

def make_data(paths_data = paths):
    X = np.empty((0, 2047))
    labels = np.empty((0, 1))
    for i in range(0, len(paths_data)):
        folder_path = paths_data[i]
        for j in range(0, 5):
            file_path = os.path.join(folder_path, f"{file_names[j]}")
            data = pd.read_csv(file_path, sep="\t")

            x = data.iloc[:, 1].values

            label = np.full((1, 1), i)

            X = np.concatenate((X, [x]), axis=0) 
            labels = np.concatenate((labels, label), axis=0) 

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    labels = labels[indices]

    return X, labels

def Norm(X, option = 'min_max'):

    if option == "min_max":
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_norm = (X - X_min) / (X_max - X_min)
        return X_norm
    # , X_min, X_max
    elif option == "z_score":
        Z_score = (X - np.mean(X)) / np.std(X)
        return Z_score
    # , np.mean(X), np.std(X)
    elif option == "l2":
        L2 = X / np.sqrt(np.sum(np.power(X, 2)))
        return L2
    elif option == "l1":
        L1 = X / np.sum(np.abs(X))
        return L1
    
def visualize(X, y, option="3d", legend = True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    fig = plt.figure()
    if option == "3d":
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for c in range(len(paths)):
        idx = np.where(y == c)[0]
        if option == "3d":
            ax.scatter3D(X[idx,3], X[idx,2], X[idx,4], c=colors[c], label=f"{labels[c]}")
        else:
            ax.scatter(X[idx,0], X[idx,1], c=colors[c], label=f"{labels[c]}")
    if legend == True:
        ax.legend()
    plt.show()

def model_predict(X, y, X_test, name, path = None):
    """ If path = None, the model will make predictions on the test set
    , otherwise it will make a prediction on a single sample """
    if path != None:
        data = pd.read_csv(path, sep="\t")
        x = data.iloc[:, 1].values
        x = Norm(x).reshape(1,X.shape[1])
    else: x = X_test

    model = name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
        # if type(name).__name__ == 'KNeighborsClassifier':
        #     num_params = model.n_neighbors
        # elif type(name).__name__ == 'DecisionTreeClassifier':
        #     num_params = model.get_n_leaves()
        # elif type(name).__name__ == 'RandomForestClassifier':
        #     num_params = model.get_params()
        # else:
        #     num_params = np.size(model.coef_)

    predict = model.predict(x)
    proba = model.predict_proba(x)
    probs = [np.round(p, 2) for p in proba]
    
    result = {"Predict": predict, "Probability": probs 
            # , "Number of parameters": num_params
              }
    return result

def calculate_time(model, X, y):
    classifier = model

    time_train_start = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X, y)
    time_train_end = time.process_time()
    time_train = time_train_end - time_train_start
    print(time_train)

def Grid_search_model(X,y):
    models = {
        'Logistic Regression': LogisticRegression(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Naive Bayes': BernoulliNB()
    }

    parameters = {
        'Logistic Regression': {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l1', 'l2'], 
                                'solver': ['liblinear']},
        'SVM': {'C': [0.01, 0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 
                'gamma': ['scale', 'auto'], 'degree': [2, 3, 4], 
                'shrinking': [True, False]},

        'KNN': {'n_neighbors': [1, 2, 3, 4], 'weights': ['uniform', 'distance'], 
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
                'p': [1, 2, 3],
                "metric": ["euclidean", "manhattan", "cosine", "correlation", "braycurtis"]},

        'Decision Tree': {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                        'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': [42]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'criterion': ['gini', 'entropy'],
                        'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4],
                        'random_state': [42]},
        'Naive Bayes': {'alpha': [0.01, 0.1, 1.0]}
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FitFailedWarning)
        warnings.filterwarnings("ignore", category=DataConversionWarning)

        for name in models:
            start_time = time.time()
            model = models[name]
            params = parameters[name]
            clf = GridSearchCV(model, params, cv=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X, y)
            end_time = time.time()
            print(f"Best parameters for {name}: {clf.best_params_}, score: {clf.best_score_ :.2f}")
            print(f"Time taken for {name}: {end_time - start_time:.2f} seconds\n")