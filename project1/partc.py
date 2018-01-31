from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import text
import numpy as np
import operator
import math

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey', 'alt.atheism', 'sci.crypt', 'sci.electronics', 'sci.med',
              'sci.space', 'soc.religion.christian', 'misc.forsale', 'talk.politics.guns',
              'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

stop_words = text.ENGLISH_STOP_WORDS
analyzer = CountVectorizer().build_analyzer()
stemmer = SnowballStemmer("english")


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def tcicf(frequency, maximum_frequency, categories, categories_per_term):
    val = ((0.5 + (0.5 * (frequency / float(maximum_frequency)))) * math.log10(categories / float(1 + categories_per_term)))
    return val


all_documents_sorted = []
for category in categories:
    twenty_train = fetch_20newsgroups(subset='train', categories=[category]).data
    data_in_category = ''
    for each_item in twenty_train:
        data_in_category = data_in_category + ' ' + each_item
    all_documents_sorted.append(data_in_category)


count_vect = CountVectorizer(analyzer='word', min_df=5, stop_words=stop_words, tokenizer=stemmed_words)
X_train_counts = count_vect.fit_transform(all_documents_sorted)

# TF_ICF starts here
X_train_counts_zero = X_train_counts.shape[0]
X_train_counts_one = X_train_counts.shape[1]

# finding maximum frequency term in each class
maximum_frequency_by_category = [0] * X_train_counts_zero
for i in range(X_train_counts_zero):
    maximum_frequency_by_category[i] = np.amax(X_train_counts[i, :])

# number of classes in which a term appears
category_count = [0] * X_train_counts_one
for i in range(X_train_counts_one):
    for j in range(X_train_counts_zero):
        if X_train_counts[j, i] == 0:
            category_count[i] += 0
        else:
            category_count[i] += 1

# initializing TC_ICF matrix with zeros

tf_icf = np.zeros((len(count_vect.get_feature_names()), X_train_counts_one))

for i in range(X_train_counts_one):
    each_row = X_train_counts[:, i].toarray()
    for j in range(X_train_counts_zero):
        tf_icf[i][j] = tcicf(each_row[j, 0], maximum_frequency_by_category[j], 20, category_count[i])

for cat in [2, 3, 13, 14]:
    TFICF = {}
    index = 0
    for eachterm in count_vect.get_feature_names():
        TFICF[eachterm] = tf_icf[index][cat]
        index += 1
    important_terms = dict(sorted(TFICF.items(), key=operator.itemgetter(1), reverse=True)[:10])
    print(important_terms.keys())
