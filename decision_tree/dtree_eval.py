import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score



def evaluatePerformance(numTrials=100):
    '''
    Evaluate the performance of decision trees,
    averaged over 1,000 trials of 10-fold cross validation
    
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree
      
    ** Note that your implementation must follow this API**
    '''
    
    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n,d = X.shape
    treeAccuracy = []
    stumpAccuracy = []
    threeAccuracy = []
    twoAccuracy = []
    fourAccuracy = []
    fiveAccuracy = []
    nineAccuracy = []
    treeAccuracyVar = []
    stumpAccuracyVar = []
    threeAccuracyVar = []
    twoAccuracyVar = []
    fourAccuracyVar = []
    fiveAccuracyVar = []
    nineAccuracyVar = []
    # shuffle the data
    idx = np.arange(n)
    for trialIdx in range(0,100):
        np.random.seed()
        np.random.shuffle(idx)
        #print idx
        X = X[idx]
        y = y[idx]
        
        # split the data
        # 27 instances in 9 of the folds and 24 in the last one
        xIdx = range(0, 266, 27)
        for ixidx in xIdx:
            
            if ixidx < 243:
                Xtrain = np.concatenate((X[0:ixidx,:], X[ixidx+27:,:]))  # train on first 100 instances
                Xtest = X[ixidx:ixidx+27,:]
                ytrain = np.concatenate((y[0:ixidx,:], y[(ixidx+27):,:]))  # test on remaining instances
                ytest = y[ixidx:(ixidx+27),:]
                for j in range(1,10):
                    XtrainVar = Xtrain[0:((24*j)+1),:]
                    ytrainVar = ytrain[0:((24*j)+1),:]
                    # train the decision tree
                    clfVar = tree.DecisionTreeClassifier()
                    clfVar = clfVar.fit(XtrainVar,ytrainVar)
                    # train the stump
                    clStumpVar = tree.DecisionTreeClassifier(max_depth = 1)
                    clStumpVar = clStumpVar.fit(XtrainVar, ytrainVar)
                    # train the 3 level
                    clThreeVar = tree.DecisionTreeClassifier(max_depth = 3)
                    clThreeVar = clThreeVar.fit(XtrainVar, ytrainVar)
                    # train the 2 level
                    clTwoVar = tree.DecisionTreeClassifier(max_depth = 2)
                    clTwoVar = clTwoVar.fit(XtrainVar, ytrainVar)
                    # train the 4 level
                    clFourVar = tree.DecisionTreeClassifier(max_depth = 4)
                    clFourVar = clFourVar.fit(XtrainVar, ytrainVar)
                    # train the 5 level
                    clFiveVar = tree.DecisionTreeClassifier(max_depth = 5)
                    clFiveVar = clFiveVar.fit(XtrainVar, ytrainVar)
                    # train the 9 level
                    clNineVar = tree.DecisionTreeClassifier(max_depth = 9)
                    clNineVar = clNineVar.fit(XtrainVar, ytrainVar)
                    # output predictions on the remaining data
                    y_pred_tree_var = clfVar.predict(Xtest)
                    y_pred_stump_var = clStumpVar.predict(Xtest)
                    y_pred_three_var = clThreeVar.predict(Xtest)
                    y_pred_two_var = clTwoVar.predict(Xtest)
                    y_pred_four_var = clFourVar.predict(Xtest)
                    y_pred_five_var = clFiveVar.predict(Xtest)
                    y_pred_nine_var = clNineVar.predict(Xtest)           
                    # compute the training accuracy of the model
                    DecisionTreeAccuracyVar = accuracy_score(ytest, y_pred_tree_var)
                    DecisionStumpAccuracyVar = accuracy_score(ytest, y_pred_stump_var)
                    DecisionThreeAccuracyVar = accuracy_score(ytest, y_pred_three_var)
                    DecisionTwoAccuracyVar = accuracy_score(ytest, y_pred_two_var)
                    DecisionFourAccuracyVar = accuracy_score(ytest, y_pred_four_var)
                    DecisionFiveAccuracyVar = accuracy_score(ytest, y_pred_five_var)
                    DecisionNineAccuracyVar = accuracy_score(ytest, y_pred_nine_var)
                    #print meanDecisionTreeAccuracy
                    treeAccuracyVar = np.append(treeAccuracyVar, DecisionTreeAccuracyVar)
                    stumpAccuracyVar = np.append(stumpAccuracyVar, DecisionStumpAccuracyVar)
                    threeAccuracyVar = np.append(threeAccuracyVar, DecisionThreeAccuracyVar)
                    twoAccuracyVar = np.append(twoAccuracyVar, DecisionTwoAccuracyVar)
                    fourAccuracyVar = np.append(fourAccuracyVar, DecisionFourAccuracyVar)
                    fiveAccuracyVar = np.append(fiveAccuracyVar, DecisionFiveAccuracyVar)
                    nineAccuracyVar = np.append(nineAccuracyVar, DecisionNineAccuracyVar)
            else:
                Xtrain = X[0:243,:]  # train on first 100 instances
                Xtest = X[243:,:]
                ytrain = y[0:243,:]  # test on remaining instances
                ytest = y[243:,:]
                for j in range(1,10):
                    XtrainVar = Xtrain[0:((24*j)+1),:]
                    ytrainVar = ytrain[0:((24*j)+1),:]
                    # train the decision tree
                    clfVar = tree.DecisionTreeClassifier()
                    clfVar = clfVar.fit(XtrainVar,ytrainVar)
                    # train the stump
                    clStumpVar = tree.DecisionTreeClassifier(max_depth = 1)
                    clStumpVar = clStumpVar.fit(XtrainVar, ytrainVar)
                    # train the 3 level
                    clThreeVar = tree.DecisionTreeClassifier(max_depth = 3)
                    clThreeVar = clThreeVar.fit(XtrainVar, ytrainVar)
                    # train the 2 level
                    clTwoVar = tree.DecisionTreeClassifier(max_depth = 2)
                    clTwoVar = clTwoVar.fit(XtrainVar, ytrainVar)
                    # train the 4 level
                    clFourVar = tree.DecisionTreeClassifier(max_depth = 4)
                    clFourVar = clFourVar.fit(XtrainVar, ytrainVar)
                    # train the 5 level
                    clFiveVar = tree.DecisionTreeClassifier(max_depth = 5)
                    clFiveVar = clFiveVar.fit(XtrainVar, ytrainVar)
                    # train the 9 level
                    clNineVar = tree.DecisionTreeClassifier(max_depth = 9)
                    clNineVar = clNineVar.fit(XtrainVar, ytrainVar)
                    # output predictions on the remaining data
                    y_pred_tree_var = clfVar.predict(Xtest)
                    y_pred_stump_var = clStumpVar.predict(Xtest)
                    y_pred_three_var = clThreeVar.predict(Xtest)
                    y_pred_two_var = clTwoVar.predict(Xtest)
                    y_pred_four_var = clFourVar.predict(Xtest)
                    y_pred_five_var = clFiveVar.predict(Xtest)
                    y_pred_nine_var = clNineVar.predict(Xtest)           
                    # compute the training accuracy of the model
                    DecisionTreeAccuracyVar = accuracy_score(ytest, y_pred_tree_var)
                    DecisionStumpAccuracyVar = accuracy_score(ytest, y_pred_stump_var)
                    DecisionThreeAccuracyVar = accuracy_score(ytest, y_pred_three_var)
                    DecisionTwoAccuracyVar = accuracy_score(ytest, y_pred_two_var)
                    DecisionFourAccuracyVar = accuracy_score(ytest, y_pred_four_var)
                    DecisionFiveAccuracyVar = accuracy_score(ytest, y_pred_five_var)
                    DecisionNineAccuracyVar = accuracy_score(ytest, y_pred_nine_var)
                    #print meanDecisionTreeAccuracy
                    treeAccuracyVar = np.append(treeAccuracyVar, DecisionTreeAccuracyVar)
                    stumpAccuracyVar = np.append(stumpAccuracyVar, DecisionStumpAccuracyVar)
                    threeAccuracyVar = np.append(threeAccuracyVar, DecisionThreeAccuracyVar)
                    twoAccuracyVar = np.append(twoAccuracyVar, DecisionTwoAccuracyVar)
                    fourAccuracyVar = np.append(fourAccuracyVar, DecisionFourAccuracyVar)
                    fiveAccuracyVar = np.append(fiveAccuracyVar, DecisionFiveAccuracyVar)
                    nineAccuracyVar = np.append(nineAccuracyVar, DecisionNineAccuracyVar)
            # train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain,ytrain)
            # train the stump
            clStump = tree.DecisionTreeClassifier(max_depth = 1)
            clStump = clStump.fit(Xtrain, ytrain)
            # train the 3 level
            clThree = tree.DecisionTreeClassifier(max_depth = 3)
            clThree = clThree.fit(Xtrain, ytrain)
            # train the 2 level
            clTwo = tree.DecisionTreeClassifier(max_depth = 2)
            clTwo = clTwo.fit(Xtrain, ytrain)
            # train the 4 level
            clFour = tree.DecisionTreeClassifier(max_depth = 4)
            clFour = clFour.fit(Xtrain, ytrain)
            # train the 5 level
            clFive = tree.DecisionTreeClassifier(max_depth = 5)
            clFive = clFive.fit(Xtrain, ytrain)
            # train the 9 level
            clNine = tree.DecisionTreeClassifier(max_depth = 9)
            clNine = clNine.fit(Xtrain, ytrain)
            # output predictions on the remaining data
            y_pred_tree = clf.predict(Xtest)
            y_pred_stump = clStump.predict(Xtest)
            y_pred_three = clThree.predict(Xtest)
            y_pred_two = clTwo.predict(Xtest)
            y_pred_four = clFour.predict(Xtest)
            y_pred_five = clFive.predict(Xtest)
            y_pred_nine = clNine.predict(Xtest)           
            # compute the training accuracy of the model
            DecisionTreeAccuracy = accuracy_score(ytest, y_pred_tree)
            DecisionStumpAccuracy = accuracy_score(ytest, y_pred_stump)
            DecisionThreeAccuracy = accuracy_score(ytest, y_pred_three)
            DecisionTwoAccuracy = accuracy_score(ytest, y_pred_two)
            DecisionFourAccuracy = accuracy_score(ytest, y_pred_four)
            DecisionFiveAccuracy = accuracy_score(ytest, y_pred_five)
            DecisionNineAccuracy = accuracy_score(ytest, y_pred_nine)
            #print meanDecisionTreeAccuracy
            treeAccuracy = np.append(treeAccuracy, DecisionTreeAccuracy)
            stumpAccuracy = np.append(stumpAccuracy, DecisionStumpAccuracy)
            threeAccuracy = np.append(threeAccuracy, DecisionThreeAccuracy)
            twoAccuracy = np.append(twoAccuracy, DecisionTwoAccuracyVar)
            fourAccuracy = np.append(fourAccuracy, DecisionFourAccuracy)
            fiveAccuracy = np.append(fiveAccuracy, DecisionFiveAccuracy)
            nineAccuracy = np.append(nineAccuracy, DecisionNineAccuracy)
            # compute the tree accuracy for learning curve
            treeAccuracyVar = np.append(treeAccuracyVar, DecisionTreeAccuracy)
            stumpAccuracyVar = np.append(stumpAccuracyVar, DecisionStumpAccuracy)
            threeAccuracyVar = np.append(threeAccuracyVar, DecisionThreeAccuracy)
            twoAccuracyVar = np.append(twoAccuracyVar, DecisionTwoAccuracy)
            fourAccuracyVar = np.append(fourAccuracyVar, DecisionFourAccuracy)
            fiveAccuracyVar = np.append(fiveAccuracyVar, DecisionFiveAccuracy)
            nineAccuracyVar = np.append(nineAccuracyVar, DecisionNineAccuracy)
    # TODO: update these statistics based on the results of your experiment
    #print treeAccuracyVar.shape
    meanDecisionTreeAccuracy = np.mean(treeAccuracy)
    stddevDecisionTreeAccuracy = np.std(treeAccuracy)
    meanDecisionStumpAccuracy = np.mean(stumpAccuracy)
    stddevDecisionStumpAccuracy = np.std(stumpAccuracy)
    meanDT3Accuracy = np.mean(threeAccuracy)
    stddevDT3Accuracy = np.std(threeAccuracy)
    meanDT2Accuracy = np.mean(twoAccuracy)
    stddevDT2Accuracy = np.std(twoAccuracy)
    meanDT4Accuracy = np.mean(threeAccuracy)
    stddevDT4Accuracy = np.std(fourAccuracy)
    meanDT5Accuracy = np.mean(fiveAccuracy)
    stddevDT5Accuracy = np.std(fiveAccuracy)
    meanDT9Accuracy = np.mean(nineAccuracy)
    stddevDT9Accuracy = np.std(nineAccuracy)
    # store the accuracies of learning curve in the array
    meanDecisionTreeAccuracyVar = np.mean(np.reshape(treeAccuracyVar, (1000, 10)), axis=0)
    stddevDecisionTreeAccuracyVar = np.std(np.reshape(treeAccuracyVar, (1000, 10)), axis=0)
    meanDecisionStumpAccuracyVar = np.mean(np.reshape(stumpAccuracyVar, (1000, 10)), axis=0)
    stddevDecisionStumpAccuracyVar = np.std(np.reshape(stumpAccuracyVar, (1000, 10)), axis=0)
    meanDT3AccuracyVar = np.mean(np.reshape(threeAccuracyVar, (1000, 10)), axis=0)
    stddevDT3AccuracyVar = np.std(np.reshape(threeAccuracyVar, (1000, 10)), axis=0)
    meanDT2AccuracyVar = np.mean(np.reshape(twoAccuracyVar, (1000, 10)), axis=0)
    stddevDT2AccuracyVar = np.std(np.reshape(twoAccuracyVar, (1000, 10)), axis=0)
    meanDT4AccuracyVar = np.mean(np.reshape(fourAccuracyVar, (1000, 10)), axis=0)
    stddevDT4AccuracyVar = np.std(np.reshape(fourAccuracyVar, (1000, 10)), axis=0)
    meanDT5AccuracyVar = np.mean(np.reshape(fiveAccuracyVar, (1000, 10)), axis=0)
    stddevDT5AccuracyVar = np.std(np.reshape(fiveAccuracyVar, (1000, 10)), axis=0)
    meanDT9AccuracyVar = np.mean(np.reshape(nineAccuracyVar, (1000, 10)), axis=0)
    stddevDT9AccuracyVar = np.std(np.reshape(nineAccuracyVar, (1000, 10)), axis=0)
    #print meanDecisionTreeAccuracyVar
    # plot learning curve
    t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #[a, b, c, d, e, f, g] = plt.plot(t, meanDecisionTreeAccuracyVar, 'r', t, meanDecisionStumpAccuracyVar, 'b', t, meanDT3AccuracyVar, 'g', t, meanDT2AccuracyVar, 'r--', t, meanDT4AccuracyVar, 'g--', t, meanDT5AccuracyVar, 'b--', t, meanDT9AccuracyVar, 'r^')
    [a, b, c, d, e, f, g] = plt.plot(t, meanDecisionTreeAccuracyVar, t, meanDecisionStumpAccuracyVar, t, meanDT3AccuracyVar, t, meanDT2AccuracyVar, t, meanDT4AccuracyVar, t, meanDT5AccuracyVar, t, meanDT9AccuracyVar)
    plt.legend([a, b, c, d, e, f, g], ["Decision Tree (without any depth restriction)", "Decision Stump", "3-level tree", "2-level tree", "4-level tree", "5-level tree", "9-level tree"])
    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    #plt.savefig('learningcurve.png')
    plt.savefig('learningcurve.pdf')
    #plt.show()

    # make certain that the return value matches the API specification
    stats = np.zeros((3,2))
    stats[0,0] = meanDecisionTreeAccuracy
    stats[0,1] = stddevDecisionTreeAccuracy
    stats[1,0] = meanDecisionStumpAccuracy
    stats[1,1] = stddevDecisionStumpAccuracy
    stats[2,0] = meanDT3Accuracy
    stats[2,1] = stddevDT3Accuracy
    return stats



# Do not modify from HERE...
if __name__ == "__main__":
    
    stats = evaluatePerformance()
    print "Decision Tree Accuracy = ", stats[0,0], " (", stats[0,1], ")"
    print "Decision Stump Accuracy = ", stats[1,0], " (", stats[1,1], ")"
    print "3-level Decision Tree = ", stats[2,0], " (", stats[2,1], ")"
# ...to HERE.
