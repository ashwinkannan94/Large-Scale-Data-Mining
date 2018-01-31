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
from sklearn.naive_bayes import GaussianNB

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


train_classification = [1] * len(computer_train.data) + [-1] * len(recreational_train.data)
test_classification = [1] * len(computer_test.data) + [-1] * len(recreational_test.data)

count_vect = CountVectorizer(analyzer='word', min_df=2, stop_words=stop_words, tokenizer=stemmed_words)
X_train_counts = count_vect.fit_transform(train_and_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# NMF
nmf = NMF(n_components=50, random_state=42)
nmf_tfidf = nmf.fit_transform(X_train_tfidf)
NMF_test_data = np.concatenate((nmf_tfidf[len(computer_train.data):(len(computer_train.data) + len(computer_test.data))],
                                nmf_tfidf[(len(computer_train.data) + len(computer_test.data) + len(recreational_train.data)):]))
NMF_train_data = np.concatenate((nmf_tfidf[0:len(computer_train.data)], nmf_tfidf[(len(computer_train.data) +
                                                                                   len(computer_test.data)):(len(computer_train.data) + len(computer_test.data) + len(recreational_train.data))]))
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(NMF_train_data, train_classification)
class_which_was_predicted = naive_bayes_classifier.predict(NMF_test_data)
actual_class_passed = test_classification
predict_probability = naive_bayes_classifier.predict_proba(NMF_test_data[:])[:, 1]
print('Accuracy for NMF is: ' + str(metrics.accuracy_score(actual_class_passed, class_which_was_predicted)))
print('Precision for NMF is: ' + str(metrics.precision_score(actual_class_passed, class_which_was_predicted, average='macro')))
print('Recall for NMF is: ' + str(metrics.recall_score(actual_class_passed, class_which_was_predicted, average='macro')))
print('Confusion matrix for NMF is: ' + str(metrics.confusion_matrix(actual_class_passed, class_which_was_predicted)))

false_positive_rate_NMF, true_positive_rate_NMF, c = roc_curve(actual_class_passed, predict_probability)
plt.figure(1)
plt.plot(false_positive_rate_NMF, true_positive_rate_NMF)
plt.plot([0, 1], [0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('Flase Positive Rate')
plt.title('ROC Curve of NMF Naive Bayes Classification With min_df=2')
plt.show()
