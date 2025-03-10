import numpy as np
import pandas as pd
from collections import defaultdict
class NaiveBayes():
    #Features in X and labels in y
    #Assume cleaned data, X is pd.dataframe
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.num_features = self.X.shape[1]
        self.classes = np.unique(self.y)
        self.num_classes = len(self.classes)
        self.priors = None
        self.likelihoods = None

    def fit(self,alpha = 1e-10):
        self.priors = self.prior()
        self.likelihoods = self.likelihood(alpha)
    
    def prior(self):
        # P(y)
        unique, counts = np.unique(self.y, return_counts=True)#1{y^(i) = c}
        return counts / counts.sum()
    
    def likelihood(self, alpha = 1e-9):
        # P(X|Y) = prod_i^C(P(x_i|Y))
        # Structure: [num_features][num_classes]
        likelihoods = [[defaultdict(lambda: alpha) for _ in range(self.num_classes)] #default laplace smoothing
                      for _ in range(self.num_features)]
        class_indices = {c: np.where(self.y == c)[0] for c in self.classes}

        # For each class
        for j, c in enumerate(self.classes):
            X_class = self.X[class_indices[c]]
            
            # Process all features for this class at once
            for i in range(self.num_features):
                unique, counts = np.unique(X_class[:, i], return_counts=True)
                prob = (counts.astype(float) + alpha) / (len(X_class) + alpha)
                
                # Update the likelihood dictionary
                for val, p in zip(unique, prob):
                    likelihoods[i][j][val] = p
        
        return likelihoods
    def predict(self,X, alpha = 1e-9):
        #P(Y|X) = P(Y)P(X|Y)/P(X)
        X = np.array(X)
        #avoid underflow
        log_posteriors = np.zeros((X.shape[0], self.num_classes))
        log_posteriors += np.log(self.priors)
        for i in range(X.shape[0]):
            for j in range(self.num_features):
                feature_val = X[i, j]
                for c in range(self.num_classes):
                    log_posteriors[i, c] += np.log(self.likelihoods[j][c].get(feature_val, alpha))
        return self.classes[np.argmax(log_posteriors, axis=1)] 
    def accuracy(self,X,y):
        y_pred =  self.predict(X) #max(P(Y_i|X))
        return np.mean(y_pred == y)
    def confusion_matrix(self,X,y):
        y_pred = self.predict(X)
        cm = np.zeros((2,2))
        for i in range(len(y)):
            cm[y[i],y_pred[i]] += 1
        return cm
    
if __name__ == "__main__":
    X_train = pd.read_csv("data\X_train.csv").to_numpy()
    y_train = pd.read_csv("data\y_train.csv").to_numpy().ravel()
    X_test = pd.read_csv("data\X_test.csv").to_numpy()
    y_test = pd.read_csv("data\y_test.csv").to_numpy().ravel()
    nb = NaiveBayes(X_train,y_train)
    nb.fit()
    print(nb.predict(X_test))
    print(nb.accuracy(X_test,y_test))

    from sklearn.naive_bayes import CategoricalNB
    classifier = CategoricalNB(alpha=1.0e-10)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    print(len(y_pred),sum(y_pred == nb.predict(X_test)))
