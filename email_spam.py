import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import keras
from nltk.corpus import stopwords
import collections
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv("spam_ham_dataset.csv", encoding='latin')
data = data.drop(columns=['Unnamed: 0'])
data = data.dropna(axis=0)
data['length'] = data['text'].apply(len)

# does preprocessing
lem = WordNetLemmatizer()
def preprocess(data):
    mail = data.lower()
    mail = re.sub("[^a-z ]", "", mail)
    mail = mail.split()
    mail = [lem.lemmatize(word) for word in mail if not word in set(stopwords.words('english'))]
    mail = " ".join(mail)
    return mail

x = data['text'].apply(preprocess)
word_count_plot(x)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(data['label'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
text2seq = tokenizer.texts_to_sequences(x)

maxseq = max([len(i) for i in text2seq])
padded = pad_sequences(text2seq, maxlen=maxseq, padding='pre')

# create training and testing sets
x_train, x_test, y_train, y_test = train_test_split(padded, y, random_state=42, test_size=0.2)

# create model
TOT_SIZE = len(tokenizer.word_index) + 1

lstm = Sequential()
lstm.add(Embedding(TOT_SIZE, 10, input_length=maxseq))
lstm.add(LSTM(100))
'''lstm.add(Dropout(0.4))
lstm.add(Dense(20, activation='relu'))
lstm.add(Dropout(0.3))'''
lstm.add(Dense(1, activation='sigmoid'))

lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm.fit(x_train, y_train, epochs=5, validation_split=0.2, batch_size=16)

# evaluate
y_pred = lstm.predict(x_test)
y_pred = (y_pred > 0.5)
print("Test Score:{:.2f}%".format(accuracy_score(y_test, y_pred)*100))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)