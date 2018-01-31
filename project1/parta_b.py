from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer

# part a
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
setting_bins = [1, 2, 3, 4, 5, 6, 7, 8]
plt.hist(twenty_train.target, bins=setting_bins, edgecolor='black', linewidth=1.2)

# plt.show()

# part b
stop_words = text.ENGLISH_STOP_WORDS
analyzer = CountVectorizer().build_analyzer()
stemmer = SnowballStemmer("english")


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


count_vect = CountVectorizer(analyzer='word', min_df=2, stop_words=stop_words, tokenizer=stemmed_words)

X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
