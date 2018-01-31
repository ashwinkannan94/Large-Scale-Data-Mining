from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.cross_validation import cross_val_predict

# part a
computer_categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                       'comp.sys.mac.hardware']
recreational_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

computer_train = fetch_20newsgroups(subset='train', categories=computer_categories, shuffle=True, random_state=42)
computer_test = fetch_20newsgroups(subset='test', categories=computer_categories, shuffle=True, random_state=42)
recreational_train = fetch_20newsgroups(subset='train', categories=recreational_categories, shuffle=True, random_state=42)
recreational_test = fetch_20newsgroups(subset='test', categories=recreational_categories, shuffle=True, random_state=42)

train_and_test = computer_train.data + computer_test.data + recreational_train.data + recreational_test.data

stop_words = text.ENGLISH_STOP_WORDS
analyzer = CountVectorizer().build_analyzer()
stemmer = SnowballStemmer("english")


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


classification = [1] * (len(computer_train.data) + len(computer_test.data)) + [-1] * (len(recreational_train.data) + len(recreational_test.data))

count_vect = CountVectorizer(analyzer='word', min_df=2, stop_words=stop_words, tokenizer=stemmed_words)
X_train_counts = count_vect.fit_transform(train_and_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# LSI
svd = TruncatedSVD(n_components=50, random_state=42)
svd_lsi_tfidf = svd.fit_transform(X_train_tfidf)

gammas = [0.001, 0.01, 0.1, 1, 10, 1000]
accuracy = []
for gamma in gammas:
    svm_classifier = svm.SVC(kernel='linear', C=gamma, random_state=42)
    svm_classifier.fit(svd_lsi_tfidf, classification)
    class_which_was_predicted = cross_val_predict(svm_classifier, svd_lsi_tfidf, classification, cv=5)
    print('Gamma Value:' + str(gamma))
    accuracy_received = metrics.accuracy_score(classification, class_which_was_predicted)
    accuracy.append(accuracy_received)
    print('Accuracy for LSI is: ' + str(accuracy_received))
    print('Precision for LSI is: ' + str(metrics.precision_score(classification, class_which_was_predicted, average='macro')))
    print('Recall for LSI is: ' + str(metrics.recall_score(classification, class_which_was_predicted, average='macro')))
    print('Confusion matrix for LSI is: ' + str(metrics.confusion_matrix(classification, class_which_was_predicted)))

maximum_accuracy = max(accuracy)
maximum_accuracy_index = accuracy.index(max(accuracy))
maximum_gamma_value = gammas[maximum_accuracy_index]
print('Maximum accuracy is for the gamma value: ' + str(maximum_gamma_value))
print('Maximum accuracy is: ' + str(maximum_accuracy))

# NMF

nmf = NMF(n_components=50, random_state=42)
nmf_tfidf = nmf.fit_transform(X_train_tfidf)
nmf_accuracy = []
for gamma in gammas:
    svm_classifier = svm.SVC(kernel='linear', C=gamma, random_state=42)
    svm_classifier.fit(nmf_tfidf, classification)
    class_which_was_predicted = cross_val_predict(svm_classifier, nmf_tfidf, classification, cv=5)
    print('Gamma Value:' + str(gamma))
    accuracy_received = metrics.accuracy_score(classification, class_which_was_predicted)
    nmf_accuracy.append(accuracy_received)
    print('Accuracy for NMF is: ' + str(accuracy_received))
    print('Precision for NMF is: ' + str(metrics.precision_score(classification, class_which_was_predicted, average='macro')))
    print('Recall for NMF is: ' + str(metrics.recall_score(classification, class_which_was_predicted, average='macro')))
    print('Confusion matrix for NMF is: ' + str(metrics.confusion_matrix(classification, class_which_was_predicted)))

maximum_nmf_accuracy = max(nmf_accuracy)
maximum_nmf_accuracy_index = nmf_accuracy.index(max(nmf_accuracy))
maximum_nmf_gamma_value = gammas[maximum_nmf_accuracy_index]
print('Maximum accuracy is for the gamma value: ' + str(maximum_nmf_gamma_value))
print('Maximum accuracy is: ' + str(maximum_nmf_accuracy))
