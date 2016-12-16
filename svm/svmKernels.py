import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):  
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return np.power((np.dot(X1, X2.T)+1),_polyDegree)


def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d1 = X1.shape
    n2,d2 = X2.shape
    if n1 > n2:
        n = n1
        m = n2
        A = X1
        B = X2
    else:
        n = n2
        m = n1
        A = X2
        B = X1
    #print "n: ", n
    #print "m: ", m
    pairwise = np.random.rand(n,m)
    for i in range(n):
        for j in range(m):
            pairwise[i][j] = np.linalg.norm(np.subtract(A[i][:], B[j][:]))
    return np.exp(-0.5*pairwise/(_gaussSigma*_gaussSigma))


def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d1 = X1.shape
    n2,d2 = X2.shape
    if n1 > n2:
        n = n1
        m = n2
        A = X1
        B = X2
    else:
        n = n2
        m = n1
        A = X2
        B = X1
    #print "n: ", n
    #print "m: ", m
    pairwise = np.random.rand(n,m)
    for i in range(n):
        for j in range(m):
            pairwise[i][j] = np.dot(A[i][:], B[:][j])/(np.linalg.norm(A[i][:])*np.linalg.norm(B[:][j]))
    return pairwise

