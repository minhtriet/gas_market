import numpy as np

np.random.seed(1)

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from util import io

news = io.load_news(embed='none')
X_train = news.head(int(len(news) * 0.6)).values
X_train = X_train.sum()
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=128000)
X_train_onehot = vectorizer.fit_transform(X_train)

word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

model = Sequential()
model.add(Embedding(len(vectorizer.get_feature_names()) + 1,
                    64,  # Embedding size
                    input_length=None, trainable=False))
model.add(LSTM(64))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
