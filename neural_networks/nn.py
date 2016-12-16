import numpy as np
from sklearn import preprocessing
from PIL import Image 

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=1.5, numEpochs=100):
        '''
        Constructor
        Arguments:
            layers - a numpy array of L-2 integers (L is # layers in the network)
            epsilon - one half the interval around zero for setting the initial weights
            learningRate - the learning rate for backpropagation
            numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs

        self.lambdaRegularization = 0.01
        self.weights = {}
        self.activationValues = {}
        self.errors = {}
        self.classes = {}

    def backPropagation(self, X, y_binarized):
        self.errors[len(self.layers)+1] = self.activationValues[len(self.layers)+1] - y_binarized
        idx = range(len(self.layers)+1)
        idx = sorted(idx, reverse=True)
        for i in idx:
            self.errors[i] = np.multiply(np.dot(self.errors[i+1],self.weights[i+1]), np.multiply(self.activationValues[i],(1-self.activationValues[i])))[:,1:]
        
        gradients = {}
        (n,d) = X.shape
    
        for j in idx:
            (l,m) = self.weights[j+1].shape
            gradients[j+1] = np.dot(np.transpose(self.errors[j+1]), self.activationValues[j])
            regularization = np.concatenate((np.zeros([l,1]), self.weights[j+1][:,1:]), axis=1)*self.lambdaRegularization
            gradients[j+1] = (gradients[j+1]/n + regularization)
            self.weights[j+1] -= self.learningRate*gradients[j+1]            


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
       '''
        (n,d) = X.shape
        self.classes = np.unique(y)
        numClasses = len(self.classes)

        oneVsMany = preprocessing.LabelBinarizer()
        oneVsMany = oneVsMany.fit(y)
        y_binarized = oneVsMany.transform(y)

        fullLayers = np.concatenate((self.layers, [numClasses]))
        fullLayers = np.concatenate(([d], fullLayers))       
        for i in range(1, len(self.layers)+2):
            self.weights[i] = np.random.random_sample([fullLayers[i], fullLayers[i-1]+1])*2.0*self.epsilon - self.epsilon

        for epoch in range(self.numEpochs):
            self.fwdPropagation(X, self.weights)
            self.backPropagation(X, y_binarized)

    def sigmoid(self, Z):
        return 1.0/(1.0+np.exp(-Z))

    def fwdPropagation(self, X, weights):
        (n,d) = X.shape
        X = np.c_[np.ones(n), X]
        self.activationValues[0] = X

        for i in range(len(self.layers)+1):
            Z = self.activationValues[i].dot(np.transpose(self.weights[i+1]))
            self.activationValues[i+1] = np.c_[np.ones(n), self.sigmoid(Z)]
        self.activationValues[len(self.layers)+1] = self.activationValues[len(self.layers)+1][:,1:]

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        (n,d) = X.shape
        self.fwdPropagation(X,self.weights)
        yPredicted = np.argmax(self.activationValues[len(self.layers)+1][:,1:], axis=1)
        return yPredicted
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        subSide = int(np.sqrt(self.weights[1][:,1:].shape[1]))
        imgSide = int(np.sqrt(self.weights[1].shape[0]))
        img = np.zeros((subSide*imgSide, subSide*imgSide))
        reshaped = {}
        for i in range(self.weights[1].shape[0]):
            scaled = (self.weights[1][i,1:] - np.min(self.weights[1][i,1:]))/np.max(self.weights[1][i,1:])
            scaled*=255
            reshaped[i] = scaled.reshape((subSide,subSide))
        idx = 0
        for j in range(imgSide):
            rowId0 = i*subSide
            rowId1 = (i+1)*subSide
            for k in range(imgSide):
                colId0 = j*subSide
                colId1 = (j+1)*subSide

                img[rowId0:rowId1, colId0:colId1] = reshaped[idx]
                idx += 1
        imShow = Image.fromarray(img.astype(uint8))
        imShow.show()
        imShow.save(filename)
