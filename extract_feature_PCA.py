from sklearn.decomposition import PCA
from utils import *
import numpy as np
from sklearn.svm import SVC

def U_for_pca(X, X_mean, n_components=1):
    X_hat = X - X_mean
    pca = PCA(n_components=n_components)
    pca.fit(X_hat)
    U = pca.components_.T
    return U

def pca(X, y, X_mean, U):
    X_hat = X - X_mean
    X_PCA = np.dot(U.T, X_hat.T).T
    print(f"Time: {calculate_time(SVC(), X_PCA, y)}")
    return X_PCA

def extract_PCA(X_train, y_train, X_mean, n_components = 3 ):
    U = U_for_pca(X_train,  X_mean, n_components = n_components)
    X_train_PCA = pca(X_train, y_train, X_mean, U)
    return X_train_PCA
