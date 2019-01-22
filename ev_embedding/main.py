from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from util import io

news = io.load_news(embed='none')
X_train = news.head(int(len(news) * 0.6)).values
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=128000)
X_train_onehot = vectorizer.fit_transform(X_train)
