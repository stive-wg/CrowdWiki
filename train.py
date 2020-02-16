import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array
from numpy import asarray
from numpy import zeros

frame = pd.read_csv("./new_data.csv")

x = frame['blurb']
y = frame['category']


label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(y.to_numpy())

Y = to_categorical(vec)

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=42)


pickle_out = open("tokenizer.pickle","wb")
pickle.dump(X_train.astype(str), pickle_out)
pickle_out.close()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train.astype(str))

X_train = tokenizer.texts_to_sequences(X_train.astype(str))
X_test = tokenizer.texts_to_sequences(X_test.astype(str))

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embeddings_dictionary = dict()
glove_file = open('./glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(1024, 5, activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(1024, 5, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(len(y.unique()), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.save("./model_v1.h5")