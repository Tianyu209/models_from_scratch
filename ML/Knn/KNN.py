import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

class KNN():
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.num_features = self.X.shape[1]
        self.classes = np.unique(self.y)
        self.num_classes = len(self.classes)
    def fit(self):
        #nothing to fit
        pass
    def predict(self, X, k=5, method='euclidean'):
        X = np.array(X)#1,d,m
        if method == "euclidean":
            dists = np.linalg.norm(X[:, np.newaxis, :] - self.X[np.newaxis, :, :],axis=2) # shape (m, n)
        else:
            dists = self.distance(X, method)
        k_indices = np.argsort(dists, axis=1)[:, :k]
        k_labels = self.y[k_indices]  # shape (m, k)
        # Majority vote: choose the most common label among the k nearest neighbors
        predictions = mode(k_labels, axis=1).mode.flatten()
        return predictions
    def distance(self, X, method = 'euclidean'):
        x = self.X.reshape(-1,self.num_features,1) #n,d,1
        if method == "l1":  
            return np.sum(np.abs(X - x), axis=1)
        elif method == "cosine": 
            return 1 - np.dot(X, x) / (np.linalg.norm(X) * np.linalg.norm(x))
        else:
            raise ValueError("Invalid distance metric")

    def accuracy(self,X,y):
        y_pred =  self.predict(X) 
        return np.mean(y_pred == y)
def catagorical_to_numeric(X):
    #cata to onehot
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    return X
def data_clean(X):
    #assign missing value to mode just for simplicity
    X = X.fillna(X.mode().iloc[0]) 
    X = catagorical_to_numeric(X)
    return X
if __name__ == '__main__':
    # Load data
    
    y = pd.read_csv(r"data\y_pred.csv",header=None).squeeze()
    X = pd.read_csv(r"data\adult.csv")
    X = data_clean(X)[X.columns[:-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, X[X.columns[-1]], test_size=0.2, random_state=3)#hold-out, fixed random seed for scoring
    
    knn = KNN(X_train.head(5000),y_train.head(5000))
    acc_test = knn.accuracy(X_test, y_test)
    print("Test Accuracy:", acc_test)

    