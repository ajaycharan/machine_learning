import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from pprint import pprint

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.metrics as smet
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

#Making instances
newsgroup_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroup_test = fetch_20newsgroups(subset='test', shuffle = True)

count_vec = CountVectorizer(lowercase = True, stop_words = 'english')

tfidf_tfm = TfidfTransformer(norm = 'l2', sublinear_tf = True)



X_train = tfidf_tfm.fit_transform(count_vec.fit_transform(newsgroup_train.data))
X_test = tfidf_tfm.transform(count_vec.transform(newsgroup_test.data))

#Multinomial NB
clf = MultinomialNB()
start = time.clock()
clf.fit(X_train, newsgroup_train.target)
stop = time.clock()
training_time_MNB = stop - start

predicted_MNB_train = clf.predict(X_train)
predicted_MNB_test = clf.predict(X_test)

accuracy_MNB_train = smet.accuracy_score(newsgroup_train.target, predicted_MNB_train)
accuracy_MNB_test = smet.accuracy_score(newsgroup_test.target, predicted_MNB_test)

precision_MNB_train = smet.precision_score(newsgroup_train.target, predicted_MNB_train, average='macro')
precision_MNB_test = smet.precision_score(newsgroup_test.target, predicted_MNB_test, average='macro')

recall_MNB_train = smet.recall_score(newsgroup_train.target, predicted_MNB_train, average='macro')
recall_MNB_test = smet.recall_score(newsgroup_test.target, predicted_MNB_test, average='macro')

#SVM cosine Kernel
svm_clf = svm.SVC(kernel = smet.pairwise.cosine_similarity, probability = True)
start = time.clock()
svm_clf.fit(X_train, newsgroup_train.target)
stop = time.clock()
training_time_SVM = stop - start

predicted_SVM_train = svm_clf.predict(X_train)
predicted_SVM_test = svm_clf.predict(X_test)

accuracy_SVM_train = smet.accuracy_score(newsgroup_train.target, predicted_SVM_train)
accuracy_SVM_test = smet.accuracy_score(newsgroup_test.target, predicted_SVM_test)

precision_SVM_train = smet.precision_score(newsgroup_train.target, predicted_SVM_train, average='macro')
precision_SVM_test = smet.precision_score(newsgroup_test.target, predicted_SVM_test, average='macro')

recall_SVM_train = smet.recall_score(newsgroup_train.target, predicted_SVM_train, average='macro')
recall_SVM_test = smet.recall_score(newsgroup_test.target, predicted_SVM_test, average='macro')

#Table
print("----------+-------------------------+-------------------------+")
print("          |       NAIVES BAYES      |  SVM with cosine Kernel |")
print("----------+-------------------------+-------------------------+")
print("Train Time|        %8.5f s       |        %8.5f s       |" % (training_time_MNB, training_time_SVM))
print("----------+-------------------------+-------------------------+")
print("          |    Train   |    Test    |    Train   |    Test    |")
print("----------+------------+------------+------------+------------+")
print("Accuracy  | %8.5f %% | %8.5f %% | %8.5f %% | %8.5f %% |" % (accuracy_MNB_train*100, accuracy_MNB_test*100, accuracy_SVM_train*100, accuracy_SVM_test*100))
print("----------+------------+------------+------------+------------+")
print("Precision | %8.5f %% | %8.5f %% | %8.5f %% | %8.5f %% |" % (precision_MNB_train*100, precision_MNB_test*100, precision_SVM_train*100, precision_SVM_test*100))
print("----------+------------+------------+------------+------------+")
print("Recall    | %8.5f %% | %8.5f %% | %8.5f %% | %8.5f %% |" % (recall_MNB_train*100, recall_MNB_test*100, recall_SVM_train*100, recall_SVM_test*100))
print("----------+------------+------------+------------+------------+")

#ROC curves
roc_classes = ['comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
roc_indices = np.asarray([newsgroup_train.target_names.index(i) for i in roc_classes])

target_binarized = label_binarize(newsgroup_test.target, classes = np.unique(newsgroup_test.target))

class_prob_NMB = clf.predict_proba(X_test)
class_prob_SVM = svm_clf.predict_proba(X_test)

MNB_fpr = {}
MNB_tpr = {}
MNB_roc_auc = {}

SVM_fpr = {}
SVM_tpr = {}
SVM_roc_auc = {}

for i in range(roc_indices.size):
	MNB_fpr[i], MNB_tpr[i], _ = roc_curve(target_binarized[:, roc_indices[i]], class_prob_NMB[:, roc_indices[i]])
	MNB_roc_auc[i] = auc(MNB_fpr[i], MNB_tpr[i]) 

	SVM_fpr[i], SVM_tpr[i], _ = roc_curve(target_binarized[:, roc_indices[i]], class_prob_SVM[:, roc_indices[i]])
	SVM_roc_auc[i] = auc(SVM_fpr[i], SVM_tpr[i])

plt.figure(1)
for i in range(roc_indices.size):
    plt.plot(MNB_fpr[i], MNB_tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'
                                   ''.format(roc_indices[i], MNB_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()

plt.figure(2)
for i in range(roc_indices.size):
    plt.plot(SVM_fpr[i], SVM_tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'
                                   ''.format(roc_indices[i], SVM_roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for SVM Classifier')
plt.legend(loc="lower right")
plt.show()
