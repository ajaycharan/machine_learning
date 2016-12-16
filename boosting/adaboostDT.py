import numpy as np
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.weights = None
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.classifiers = None
        self.classifierWeights = None
        self.predicted = None
        self.numClasses = 0
        self.weightMat = None
            
    def loopIter(self, X,y):
        numInstances = y.shape[0]
        #print 'shape X: ', X.shape
        self.weights = np.transpose((1./numInstances)*np.ones(numInstances))

        self.classifierWeights = np.transpose(np.zeros(self.numBoostingIters))
        dClassi = []
        self.classifiers = []
        self.numClasses = len(np.unique(y))
        #for l in range(numInstances):
        #    self.weights.append((1./numInstances))
        self.weightMat = np.zeros((self.numBoostingIters, numInstances))
        for i in range(self.numBoostingIters):
            #X = np.multiply(X, self.weights)
            self.weightMat[i,:] = self.weights
            X = np.transpose(np.multiply(np.transpose(X), np.transpose(self.weights)))
            #for j in range(X.shape[1]):
            #    X[:][j] = np.multiply(self.weights, X[:][j])
            alpha, dtclf = self.weightUpdate(X, y)
            #self.classifierWeights.append(alpha)
            if alpha == None:
                alpha = 0
                self.classifiers.append(self.classifiers[0])
            else:
                self.classifierWeights[i] = alpha
                #dClassi.append(dtclf)
                self.classifiers.append(dtclf)
        #self.classifiers = dClassi
        print 'classifier weight shape: ', self.classifierWeights.shape
        #print 'predicted shape: ', self.classifiers[i].predict(X).shape

    def dtree(self, X, y):
        dtclf = tree.DecisionTreeClassifier(max_depth = self.maxTreeDepth)
        dtclf = dtclf.fit(X, y)
        predictions = dtclf.predict(X)
        #print 'decsion tree predictions: ', predictions
        #print 'actual values: ', y
        misclassificatons = abs(np.subtract(predictions, y))
        return misclassificatons, dtclf

    def weightUpdate(self, X, y):
        misclassificatons, dtclf = self.dtree(X, y)
        #print 'misclassifications: ', misclassificatons
        
        erroneousIndices = np.transpose(np.nonzero(misclassificatons))
        #print 'error indices: ', np.transpose(erroneousIndices)
        cnt = 0
        erroneousWieghts = np.zeros(len(erroneousIndices))
        for k in range(len(erroneousIndices)):
            erroneousWieghts[cnt] = self.weights[erroneousIndices[k]]
            cnt +=1            
        print 'erroneousWieghts: ', erroneousWieghts
        #error = np.true_divide(len(erroneousIndices),len(y))
        error = np.sum(erroneousWieghts)/np.sum(self.weights)
        print 'error: ', error
        if (error >= 0.5) or (error < 0) :
            return None, None
        
        #print 'error: ', error
        alpha = 0.5*(np.log((1. - error)/error) + np.log(self.numClasses))
        #self.erroneousWieghts *= np.exp(-alpha)
        #print 'Old Wieghts: ', self.weights
        for l in range(len(erroneousIndices)):
            self.weights[erroneousIndices[l]] *= np.exp(alpha)    #weight update
        self.weights *= 1./(np.sum(self.weights))    #normalize
        
        #X = weights*X[erroneousIndices][:]
        #return X
        return alpha, dtclf

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
        self.loopIter(X,y)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        n = X.shape[0]
        print 'n: ', n
        #self.predicted = []
        self.predicted = np.empty([n,self.numBoostingIters])
        for i in range(self.numBoostingIters):
            predictedY = (self.classifiers[i].predict(X))
            #weight = self.classifierWeights[i]
            #print 'predicted : ', predictedY
            weight = self.weightMat[i,:]
            self.predicted[:,i] = (np.multiply(weight, predictedY))
        #self.predicted = np.transpose(self.predicted)
        #print 'predicted shape: ', self.predicted.shape
        y = np.around(np.transpose(np.divide(np.sum(self.predicted, axis=1),np.sum(self.weightMat, axis = 0))))
        print 'y: ', y
        return y
