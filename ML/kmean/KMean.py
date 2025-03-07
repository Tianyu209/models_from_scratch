import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansPlusPlus():
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X):
        self.X = np.array(X)
        self.n_samples, self.n_features = self.X.shape
        self.centroids = self._init_centroids_plus_plus()
        for i in range(self.max_iter):
            distances = self._compute_distance(self.X, self.centroids)
            self.labels = np.argmin(distances, axis=1)#n,c
            new_centroids = np.zeros((self.n_clusters, self.n_features))
            for j in range(self.n_clusters):
                if np.sum(self.labels == j) > 0:
                    #update centroids
                    new_centroids[j] = np.mean(self.X[self.labels == j], axis=0)
                else:
                    new_centroids[j] = self.X[np.random.randint(self.n_samples)]
            
            # stop if Converge else continue
            if np.sum((new_centroids - self.centroids) ** 2) < self.tol:
                break
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        X = np.array(X)
        distances = self._compute_distance(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _init_centroids_plus_plus(self):
        np.random.seed(self.random_state)
        centroids = np.zeros((self.n_clusters, self.n_features))
        idx = np.random.randint(self.n_samples)
        centroids[0] = self.X[idx]
        #kmean++
        for i in range(1, self.n_clusters):
            # min(d(x^(i),C^(j)))
            distances = np.min(self._compute_distance(self.X, centroids[:i]), axis=1)#n,
            # 
            probs = distances ** 2
            probs = probs / np.sum(probs)# softmax
            cumulative_probs = np.cumsum(probs)#CDF, [0,1]
            r = np.random.rand()

            for index, p in enumerate(cumulative_probs):
                #P_x = d_i^2/sum_j d_j^2
                if r < p:
                    centroids[i] = self.X[index]
                    break
        
        return centroids
    
    def _compute_distance(self, X, centroids):
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        
        for i, centroid in enumerate(centroids):
            #distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return distances #n,c
    def accuracy(self, X, y):
        #labels may be different
        return adjusted_rand_score(self.predict(X), y)
    def visualize(self):
        import matplotlib.pyplot as plt
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='red')
        plt.show()
def Test(X_train, X_test, y_test):
    kmeans = KMeans(n_clusters=3, random_state=3)
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    return adjusted_rand_score(y_pred,y_test )
def normalize( X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)# hold-out
    kmeans = KMeansPlusPlus(n_clusters=3, random_state=3)
    # X_train = normalize(X_train)
    # X_test = normalize(X_test)

    kmeans.fit(X_train)
    acc = kmeans.accuracy(X_test,y_test)
    kmeans.visualize()
    

    print("\nAccuracy:", acc)
    print("\n Kmean from sklearn: ",Test(X_train, X_test, y_test))
    
 