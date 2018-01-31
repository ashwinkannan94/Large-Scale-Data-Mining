from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn import metrics
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


count_vect = CountVectorizer(analyzer='word', min_df=5, stop_words=stop_words, tokenizer=stemmed_words)
X_train_counts = count_vect.fit_transform(train_and_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print('LSI SVM')
svd = TruncatedSVD(n_components=50, random_state=42)
svd_lsi_tfidf = svd.fit_transform(X_train_tfidf)

LSI_train_data = svd_lsi_tfidf[0:len(train.data)]
LSI_test_data = svd_lsi_tfidf[len(train.data):]

print('One VS One Classifier LSI SVM')
onevsonesvm_classifier = OneVsOneClassifier(svm.SVC(kernel='linear'))
onevsonesvm_classifier.fit(LSI_train_data, train.target)
class_which_was_predicted = onevsonesvm_classifier.predict(LSI_test_data)
print('Accuracy for onevsone SVM classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsone SVM classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsone SVM classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsone SVM classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))

print('One VS Rest Classifier LSI SVM')
onevsrestsvm_classifier = OneVsRestClassifier(svm.SVC(kernel='linear'))
onevsrestsvm_classifier.fit(LSI_train_data, train.target)
class_which_was_predicted = onevsrestsvm_classifier.predict(LSI_test_data)
print('Accuracy for onevsrest SVM classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsrest SVM classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsrest SVM classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsrest SVM classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))

# NMF
print('NMF SVM')
nmf = NMF(n_components=50, random_state=42)
nmf_tfidf = nmf.fit_transform(X_train_tfidf)

NMF_train_data = nmf_tfidf[0:len(train.data)]
NMF_test_data = nmf_tfidf[len(train.data):]

print('One VS One Classifier NMF SVM')
onevsonesvm_classifier = OneVsOneClassifier(svm.SVC(kernel='linear'))
onevsonesvm_classifier.fit(NMF_train_data, train.target)
class_which_was_predicted = onevsonesvm_classifier.predict(NMF_test_data)
print('Accuracy for onevsone SVM classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsone SVM classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsone SVM classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsone SVM classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))

print('One VS Rest Classifier NMF SVM')
onevsrestsvm_classifier = OneVsRestClassifier(svm.SVC(kernel='linear'))
onevsrestsvm_classifier.fit(NMF_train_data, train.target)
class_which_was_predicted = onevsrestsvm_classifier.predict(NMF_test_data)
print('Accuracy for onevsrest SVM classifier is: ' + str(metrics.accuracy_score(test.target, class_which_was_predicted)))
print('Precision for onevsrest SVM classifier is: ' + str(metrics.precision_score(test.target, class_which_was_predicted, average='macro')))
print('Recall for onevsrest SVM classifier is: ' + str(metrics.recall_score(test.target, class_which_was_predicted, average='macro')))
print('Confusion matrix for onevsrest SVM classifier is: ' + str(metrics.confusion_matrix(test.target, class_which_was_predicted)))
