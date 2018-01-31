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
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import svm

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=categories)
test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42, categories=categories)
train_and_test = train.data + test.data

stop_words = text.ENGLISH_STOP_WORDS
analyzer = CountVectorizer().build_analyzer()
stemmer = SnowballStemmer("english")


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


count_vect = CountVectorizer(analyzer='word', min_df=2, stop_words=stop_words, tokenizer=stemmed_words)
X_train_counts = count_vect.fit_transform(train_and_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

nmf = NMF(n_components=50, random_state=42)
nmf_tfidf = nmf.fit_transform(X_train_tfidf)

NMF_train_data = nmf_tfidf[0:len(train.data)]
NMF_test_data = nmf_tfidf[len(train.data):]

print('One VS One Classifier Naive Bayes')
onevsonenb_classifier = OneVsOneClassifier(GaussianNB())
onevsonenb_classifier.fit(NMF_train_data, train.target)
class_which_was_predicted = onevsonenb_classifier.predict(NMF_test_data)
print('Accuracy for onevsone naive bayes classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsone naive bayes classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsone naive bayes classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsone naive bayes classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))


print('One VS Many Classifier Naive Bayes')
onevsrestnb_classifier = OneVsRestClassifier(GaussianNB())
onevsrestnb_classifier.fit(NMF_train_data, train.target)
class_which_was_predicted = onevsrestnb_classifier.predict(NMF_test_data)
print('Accuracy for onevsrest naive bayes classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsrest naive bayes classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsrest naive bayes classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsrest naive bayes classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))
