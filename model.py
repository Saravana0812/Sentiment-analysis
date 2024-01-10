# Utilities
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import SimpleRNN,Dense,Activation,Bidirectional,GlobalMaxPool1D
from tensorflow.keras.layers import LSTM, Dropout
from keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay , confusion_matrix
from predict import preprocess_reviews
import matplotlib.pyplot as plt

dataset = pd.read_csv('./movie.csv')
dataset['cleaned_review'] = dataset.text.apply(preprocess_reviews)


X_data, y_data = np.array(dataset['cleaned_review']), np.array(dataset['label'])
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.05, random_state = 0)
print('Data Split done.')


Embedding_dimensions = 100
# Creating Word2Vec training dataset.
Word2vec_train_data = list(map(lambda x: x.split(), X_train))
# Defining the model and training it.
word2vec_model = Word2Vec(Word2vec_train_data,
                 vector_size=Embedding_dimensions,
                 workers=8,
                 min_count=5)
print("Vocabulary Length:", len(word2vec_model.wv.key_to_index))



input_length = 750
vocab_length = 35000
tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)
X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test  = pad_sequences(tokenizer.texts_to_sequences(X_test) , maxlen=input_length)

print("X_train.shape:", X_train.shape)
print("X_test.shape :", X_test.shape)
embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)
print("Embedding Matrix Shape:", embedding_matrix.shape)

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]


def getModel3():
    embedding_layer = Embedding(input_dim = vocab_length,
                                output_dim = Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(50,return_sequences=True)),
        GlobalMaxPool1D(),
        Dense(40, activation="relu"),
        Dropout(0.5),
        Dense(20, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ],
    name="Sentiment_Model_LSTM_Bidirectional")
    return model
training_model3 = getModel3()
print(training_model3.summary())

optimizer = SGD(learning_rate=0.01,
    momentum=0.9,
    nesterov=False,
    weight_decay=0.01,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    name="SGD"
)

training_model3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history3 = training_model3.fit(
    X_train, y_train,
    batch_size=512,
    epochs=20,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)

def get_labels(y_pred,y_test):
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    return labels

y_pred3 = training_model3.predict(X_test)

# Converting prediction to reflect the sentiment predicted.
y_pred3 = np.where(y_pred3>=0.5, 1, 0)
l3 = get_labels(y_pred3,y_test)

# Printing out the Evaluation metrics.
fig, ax = plt.subplots(1,3,figsize=(15,8))
ax[2].set_title("LSTM Bidirectional")
ConfusionMatrixDisplay.from_predictions(y_test,y_pred3,cmap="Blues" ,ax = ax[2])
plt.show()