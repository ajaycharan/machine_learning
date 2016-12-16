import numpy as np
import sklearn as skl

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.classes = None
        self.Cond_Probs = None
        self.class_Probs = None

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        if self.classes == None:
            self.classes = np.unique(y)
        num_classes = self.classes.size
        if self.Cond_Probs == None:
            self.Cond_Probs = np.zeros([num_classes, d])
        if self.class_Probs == None:    
            self.class_Probs = np.zeros(num_classes)

        if (self.useLaplaceSmoothing):
            for i in range(num_classes):
                # i_class_elements = X[np.logical_or.reduce([X == x for x in self.classes[i]])]   # This will be useful if we need more than one class.
                i_class_elements = X[np.logical_or.reduce([y == self.classes[i]])]
                self.class_Probs[i] = i_class_elements.shape[0] / float(n)
                self.Cond_Probs[i,:] = (1.0 + np.sum(i_class_elements, axis = 0)) / (d + np.sum(i_class_elements))
        else:
            for i in range(num_classes):
                i_class_elements = X[np.logical_or.reduce([y == self.classes[i]])]
                self.class_Probs[i] = i_class_elements.shape[0] / float(n)
                self.Cond_Probs[i,:] = np.sum(i_class_elements, axis = 0) / np.sum(i_class_elements)

        # print np.sum(self.Cond_Probs, axis=1)        # Checked that all the row conditional probabilities add up to 1.
        # print self.Cond_Probs.shape

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # print X.shape
        Pred_class_Probs = self.predictProbs(X)
        index = np.argmax(Pred_class_Probs, axis=1)
        # print index.shape
        return self.classes[index]
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''

        Pred_class_Probs = np.asmatrix(X) * np.asmatrix(np.log(self.Cond_Probs).T)
        # print Pred_class_Probs[0,:]
        Pred_class_Probs += np.log(self.class_Probs)
        Pred_class_Probs -= np.mean(Pred_class_Probs)
        # print Pred_class_Probs[0,:]
        Pred_class_Probs = np.exp(Pred_class_Probs)
        l1_norm = skl.preprocessing.normalize(Pred_class_Probs, norm = 'l1', axis = 1)
        return l1_norm

        
        
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing
        self.classes = None             # Vector with all the classes
        self.Cond_Probs = None          # Vector of number of classes by features of conditional probabilities
        self.class_Probs = None         # Absolute probability of a class
        self.instances = None           # Total number of instances that have been passed
        self.class_count = None         # Vector that stores all the instances that have been passed for a certain class
        self.feature_count = None       # Vector that accumulates the total count for each feature
        self.total_sum = None           # Keep track of all the "words" in the dictionary for classes

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        # print ""
        # print "============= NEW CALL TO FIT ================"
        n,d = X.shape
        # print "X shape: ", X.shape
        # print y

        # Keep track of how many total instances we have
        if self.instances == None:
            self.instances = n
        else:
            self.instances += n
        # print "Total Instances Processed: ", self.instances

        # Keep track of how many total classes we have
        if self.classes == None:
            self.classes = np.unique(y)
            diff = 0
            # print "First Time Classes: ", self.classes
        else:
            classes = np.unique(np.r_[self.classes, y])
            diff = classes.size - self.classes.size
            if diff > 0:
                new_classes_bool = np.in1d(classes, self.classes)                           # Bool vector of elements in A, that are also in B
                new_classes = classes[np.logical_or.reduce([new_classes_bool == False])]    # Extract the values of the elements that are not repeated
                # print "New Classes to Add: ", new_classes
                # self.classes = np.c_[self.classes, new_classes][0]                          # Concatenate the value of those new classes
                self.classes = np.concatenate((self.classes, new_classes), axis=1)
                # print "Classes: ", self.classes

        num_classes = self.classes.size
        # print "Number of classes: ", num_classes

        # Keep track of how many instances of each class we have
        if self.class_count == None:
            self.class_count = np.zeros(num_classes)
            # print type(self.class_count)
            
        # Initialize only one time the Conditional Probability Matrix, and then upgrade correspondingly
        if self.Cond_Probs == None:
            self.Cond_Probs = np.zeros([num_classes, d])
            
        # Keep track of the count of all the features for every class
        if self.feature_count == None:
            self.feature_count = np.zeros([num_classes, d])

        # Keep track of the sum of words per class
        if self.total_sum == None:
            self.total_sum = np.zeros(num_classes)

        if self.class_Probs == None:
            self.class_Probs = np.zeros(num_classes)

        # Update all the vectors and matrices with the new dimensions
        if diff > 0:
            self.class_count = np.concatenate((self.class_count, np.zeros(diff)), axis=1)
            self.total_sum = np.concatenate((self.total_sum, np.zeros(diff)),axis=1)
            self.Cond_Probs = np.concatenate((self.Cond_Probs, np.zeros([diff, d])), axis=0)
            self.feature_count = np.concatenate((self.feature_count, np.zeros([diff, d])), axis=0)
            self.class_Probs = np.concatenate((self.class_Probs, np.zeros(diff)), axis=1)


        if (self.useLaplaceSmoothing):
            for i in range(num_classes):
                i_class_elements = X[np.logical_or.reduce([y == self.classes[i]])]
                if i_class_elements.shape[0] > 0:
                    # Keeps track of how many instances of each class have been present
                    self.class_count[i] += i_class_elements.shape[0]
                    self.class_Probs[i] = self.class_count[i] / float(self.instances)
                    # Keep track of all the counts for each instance
                    self.feature_count[i,:] += np.sum(i_class_elements, axis = 0)
                    # Calculate conditional Probabilities according to the historic count
                    self.total_sum[i] += np.sum(i_class_elements)   # Updates sum of total words
                    self.Cond_Probs[i,:] = (1.0 + self.feature_count[i,:]) / (d + self.total_sum[i])
        else:
            for i in range(num_classes):
                i_class_elements = X[np.logical_or.reduce([y == self.classes[i]])]
                self.class_Probs[i] = i_class_elements.shape[0] / float(n)
                self.Cond_Probs[i,:] = np.sum(i_class_elements, axis = 0) / np.sum(i_class_elements)

        # print np.sum(self.Cond_Probs, axis=1)        # Checked that all the row conditional probabilities add up to 1.
        # print "Size of Conditional Probability matrix: ", self.Cond_Probs.shape 
        # print "Class Count: ", self.class_count, "Total count: ", np.sum(self.class_count)
        # print "Total Sum: ", self.total_sum
        # print "Conditional Probability Shape: ", self.Cond_Probs.shape
        # print "Feature Count: ", self.feature_count.shape
        # print "Class Probabilities: ", self.class_Probs

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        # print X.shape
        Pred_class_Probs = self.predictProbs(X)
        index = np.argmax(Pred_class_Probs, axis=1)
        # print index.shape
        # Order the classes so that they correspond with the changes made to predictProbs
        ordered_classes = self.classes[np.argsort(self.classes)]
        return ordered_classes[index]

    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''

        Pred_class_Probs = np.asmatrix(X) * np.asmatrix(np.log(self.Cond_Probs).T)
        # print Pred_class_Probs[0,:]
        Pred_class_Probs += np.log(self.class_Probs)
        Pred_class_Probs -= np.mean(Pred_class_Probs)
        # print Pred_class_Probs[0,:]
        Pred_class_Probs = np.exp(Pred_class_Probs)
        l1_norm = skl.preprocessing.normalize(Pred_class_Probs, norm = 'l1', axis = 1)
        # print np.sum(l1_norm, axis = 1)
        # Order them in ascending order and that should fix the test script
        return l1_norm[:, np.argsort(self.classes) ]

        # Pred_class_Probs = np.asmatrix(X) * np.asmatrix(np.log(self.Cond_Probs).T)
        # Pred_class_Probs += np.log(self.class_Probs)
# return Pred_class_Probs
