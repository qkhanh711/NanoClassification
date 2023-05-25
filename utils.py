import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
from sklearn.exceptions import FitFailedWarning, DataConversionWarning
import time

from sklearn.metrics import precision_recall_fscore_support, accuracy_score , confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

import warnings
from sklearn.exceptions import FitFailedWarning, DataConversionWarning

import scikitplot as skplt
import pickle
from os import path
import os
import seaborn as sn

paths = ["Data for Nano-AI/4-nitrophenol", "Data for Nano-AI/Carbaryl", "Data for Nano-AI/Chloramphenicol", "Data for Nano-AI/Congo Red", "Data for Nano-AI/Crystal Violet", 
         "Data for Nano-AI/Glyphosate", "Data for Nano-AI/Methylene Blue", "Data for Nano-AI/Thiram", "Data for Nano-AI/Tricyclazole", "Data for Nano-AI/Urea"]
labels = ["4_nitrophenol", "Carbaryl", "Chloramphenicol", "Congo red", "Crystal Violet", "Glyphosate", "Methylene", "Thiram", "Tricylazole", "Urea"]

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

def make_data(paths_data = paths, num_features=None):
    """Number of features = 34 to train
                            10 to visualize"""
    X = np.empty((0, num_features))
    labels = np.empty((0, 1))
    for i in range(0, len(paths_data)):
        folder_path = paths_data[i]
        for j in range(0, 5):
            file_path = os.path.join(folder_path, f"{file_names[j]}")
            data = pd.read_csv(file_path, sep="\t")

            if num_features == 10: 
                x = data.iloc[[1885, 1391, 1670, 1407, 1421, 1577, 1878, 1512, 1892, 1596], 1].values
            else:
                x = data.iloc[[1885, 1856, 1463, 1422,
                            1888, 1860, 1473, 1356,
                            1506, 1410, 1266, 921,
                            1782, 1513, 1393, 1270,
                                  1438, 1641, 1503,
                            1872, 1713, 1666, 1576,
                            1876, 1849, 1382, 1255,
                            1878, 1820, 1520, 1395,
                                  1608, 1428, 
                            1596], 1].values    
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
        # X = np.maximum(X, 0)
        X_norm = (X - X_min) / (X_max - X_min)
        return X_norm
    elif option == "z_score":
        X_mean = np.mean(X)
        X_std = np.std(X)
        Z_score = (X - X_mean) / X_std
        return Z_score
    
def visualize(X, y, option="3d", eval= 0, azim = 0, legend = True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
    fig = plt.figure()
    if option == "3d":
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    for c in range(len(paths)):
        idx = np.where(y == c)[0]
        if option == "3d":
            ax.scatter3D(X[idx,0], X[idx,1], X[idx,2], c=colors[c], label=f"{labels[c]}")
            ax.view_init(elev=eval, azim=azim)
        else:
            ax.scatter(X[idx,0], X[idx,1], c=colors[c], label=f"{labels[c]}")
    if legend == True:
        ax.legend()
    
    plt.show()

def model_predict(X_test, y_test, name, path = None, print_eval = True):
    print("Testing with " + type(name).__name__)
    """ If path = None, the model will make predictions on the test set
    , otherwise it will make a prediction on a single sample """
    if path != None:
        data = pd.read_csv(path, sep="\t")
        x = data.iloc[[1885, 1856, 1463, 1422,
                            1888, 1860, 1473, 1356,
                            1506, 1410, 1266, 921,
                            1782, 1513, 1393, 1270,
                                  1438, 1641, 1503,
                            1872, 1713, 1666, 1576,
                            1876, 1849, 1382, 1255,
                            1878, 1820, 1520, 1395,
                                  1608, 1428, 
                            1596], 1].values
        x = Norm(x).reshape(1,X_test.shape[1])

    else: x = X_test

    model = name

    predict = model.predict(x)
    proba = model.predict_proba(x)
    probs = [np.round(p, 2) for p in proba]
    
    result = {"Predict": predict, "Class" : [labels[int(p)] for p in predict], "Probability": probs}
    if path == None and print_eval == True:
        evaluate_model(name, X_test, y_test, labels)
    return result

def calculate_time(model, X, y):
    classifier = model

    time_train_start = time.process_time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier.fit(X, y)
    time_train_end = time.process_time()
    time_train = time_train_end - time_train_start
    return time_train

def Grid_search_model(X,y, cv = 2):
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
                                'solver': ['lbfgs','liblinear']},

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
            clf = GridSearchCV(model, params, cv=cv)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X, y)
            end_time = time.time()
            print(f"Best parameters for {name}: {clf.best_params_}, score: {clf.best_score_ :.2f}")
            print(f"Time taken for {name}: {end_time - start_time:.2f} seconds\n")

def evaluate_model(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    
    np.seterr(divide='ignore', invalid='ignore') 
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    cm_percent = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    
    plt.figure(figsize=(10.2, 7))
    sn.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    plt.xlabel("Predicted (%)")
    plt.ylabel("True label")
    plt.savefig(f"./plots/{type(model).__name__}.png")
    
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1-score: {:.4f}".format(f1_score))
   
def save_model(file_name, model):
    if not path.isfile(file_name):
        # saving the trained model to disk
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
        print(f"Saved model '{file_name}' to disk")
    else:
        print(f"Model '{file_name}' already saved")

def calculate_average_probabilities(X_test, y_test, model):
    result = model_predict(X_test, y_test, model, print_eval = False)
    labels = result["Class"]
    probabilities = result["Probability"]
    
    average_probabilities = []
    for label in set(labels):
        indices = [i for i, value in enumerate(labels) if value == label]
        label_probabilities = np.mean([probabilities[i] for i in indices], axis=0)
        average_probabilities.append(label_probabilities)
    sn.heatmap(average_probabilities, annot=True, fmt=".1f", cmap="Blues")
    return average_probabilities
