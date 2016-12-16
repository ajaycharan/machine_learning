import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        #theta = np.zeros(shape=(d,1))
        print theta

        for i in xrange(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta
            # TODO:  add update equation here
            predictedY = np.dot(X, theta)
            error = np.dot(np.transpose(X), np.subtract(predictedY, y))
            tmp = np.subtract(theta , ((1.0/n)*self.alpha*error))
            theta = tmp
        return theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        # TODO: add objective (cost) equation here
        n = y.size
        predictedY = np.dot(X, theta)
        error = np.power((predictedY - y), 2)
        J = (1.0/(2*n))*error.sum()
        return J

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        if self.theta==None:
            self.theta = np.matrix(np.zeros((d,1)))
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # TODO:  add prediction function here
        y = np.dot(X, self.theta)
        return y
