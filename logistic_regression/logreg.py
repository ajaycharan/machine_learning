import numpy as np
from numpy import linalg as LA


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None
        self.JHist = None
        self.xMean = None
        self.xStd = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        #m = len(y)
        
        yLogPredicted = np.log(self.sigmoid((np.dot(X, theta))))
        #oneVec = np.ones((m,1))
        error = -1* (np.multiply(y, yLogPredicted) + np.multiply((1- y), (1 - yLogPredicted)))
        J = error.sum() + regLambda/2.0*LA.norm(theta)
        return J
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        XT = np.transpose(X)
        self.JHist = []
        lambdaVec = regLambda*np.ones((d,1))
        lambdaVec[0] = 0.0
        for i in xrange(self.maxNumIters):
            self.JHist.append( (self.computeCost(theta, X, y, regLambda), theta) )
            print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta ############
            '''
            if i > 0:
                print "the difference in gradient fn is: ", np.subtract(theta, self.JHist[i-1][1])
                if self.hasConverged() == True:
                    break
                else:
                    # update equation
                    predictedY = self.sigmoid(np.dot(X, theta))
                    theta -= self.alpha*(np.dot(XT,(predictedY-y)) + np.multiply(lambdaVec, theta))
            else:
            '''
            # update equation
            predictedY = self.sigmoid(np.dot(X, theta))
            error = np.dot(XT,(predictedY-y)) + np.multiply(lambdaVec, theta)
            tmp = np.subtract(theta, self.alpha*error)
            theta = tmp
            if LA.norm(self.alpha*(np.dot(XT,(predictedY-y)) + np.multiply(lambdaVec, theta))) <= self.epsilon:
                i = self.maxNumIters
                break
        return theta
    
    def sigmoid(self, Z):
        '''
        Computers the sigmoid function 1/(1+exp(-z))
        '''
        sig = 1.0/(1.0 + np.exp(-1.0*Z))
        return sig
    '''
    def hasConverged(self):
        #difference = np.subtract(self.JHist[-1][1], self.JHist[-2][1])
        #difference = (self.theta) - (self.JHist[-2][1])
        #print "difference in theta: ", difference #######
        if LA.norm(np.subtract((self.theta), (self.JHist[-2][1]))) < self.epsilon:
            #print "Norm of difference: ", LA.norm(difference) #######
            result = True
        else:
            result = False
        return result
    '''
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        self.xMean = np.mean(X)
        self.xStd = np.std(X)
        X = (X - self.xMean)/self.xStd  
        X = np.c_[np.ones((n,1)),X]
        n,d = X.shape
        if self.theta==None:
            self.theta = np.random.rand(d,1)
            self.theta -= np.mean(self.theta)
            print "theta initially: ", self.theta ###################
        self.theta = self.computeGradient(self.theta, X, y, self.regLambda)



    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d = X.shape
        print "n is : ", n ###################
        X = (X - self.xMean)/self.xStd
        X = np.c_[np.ones((n,1)),X]
        predictedY = self.sigmoid(np.dot(X,self.theta))
        y = np.around(predictedY)
        print y[:72] #####################
        return y
