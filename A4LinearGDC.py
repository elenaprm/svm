from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from numpy.linalg import norm

class LinearGDC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)
        
    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)

        # Convert {+1,0} to {+1,-1}
        y_ = np.where(y<=0, -1, 1)

        n_samples, n_features = X.shape
        # Let's initialize w with n random values here, set b as a random value also
        w = np.random.rand(n_features)
        b = np.random.rand()
        # create a container to save the value of Js in each epoch
        self.Js=[0 for i in range(self.n_epochs)]

        lamda = 0.01
        # Training
        for epoch in range(self.n_epochs):
            dist = y_ * (np.sum(w * X, axis=1) + b)
            condition = 1 > dist
            y_wrong = y_[condition]
            X_wrong = X[condition]

            # compute J and the derivative Js
            # save J
            self.Js[epoch] = self.C * (1 - dist[condition]).sum() + lamda * norm(w, 2)
            w_gradient_vector = 2 * lamda * w - self.C * np.dot(y_wrong, X_wrong)
            b_derivative = self.C * -y_wrong.sum()

            # update w and b
            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])

        return self

    # output the distance between X and the boundary
    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    # output the predicted class
    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)

    # plot the value of the cost function Js for each epoch
    def plotJs(self):
        plt.ylim(0, 200)
        plt.plot(range(self.n_epochs), self.Js)
        plt.title("Value of the cost function Js for each epoch")
        plt.xlabel("num epochs")
        plt.ylabel("cost functions Js")
        plt.show()

# We will use iris dataset in this example
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1) # Iris virginica

# you can pick another value for C
C=2
svm_clf = LinearGDC(C=C, eta0 = 10, eta_d = 1000, n_epochs=60000, random_state=2)
svm_clf.fit(X, y)
svm_clf.plotJs()
