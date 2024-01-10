import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import pickle
import pandas as pd
import streamlit as st
from keras import backend as K
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
input_length = 750
vocab_length = 35000
 
new_model = tf.keras.models.load_model('./my_model_new.keras')

# load the sentiment analysis model
@st.cache(allow_output_mutation=True)
def Load_model():
    with tf.device('/cpu:0'):
        # new_model = load_model('test_model.h5')
        model = tf.keras.models.load_model('./my_model_new.keras')
    print(model.summary()) # included making it visible when the model is reloaded
    session = K.get_session()
    return model, session


contractions = pd.read_csv('./contractions.csv', index_col='Contraction')
contractions.index = contractions.index.str.lower()
contractions.Meaning = contractions.Meaning.str.lower()
contractions_dict = contractions.to_dict()['Meaning']

# Defining regex patterns.
linebreaks        = "<br /><br />"
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_reviews(review):

    review = review.lower()

    review = re.sub(linebreaks," ",review)
    # Replace 3 or more consecutive letters by 2 letter.
    review = re.sub(sequencePattern, seqReplacePattern, review)

    # Replace all emojis.
    review = re.sub(r'<3', '<heart>', review)
    review = re.sub(smileemoji, '<smile>', review)
    review = re.sub(sademoji, '<sadface>', review)
    review = re.sub(neutralemoji, '<neutralface>', review)
    review = re.sub(lolemoji, '<lolface>', review)

    for contraction, replacement in contractions_dict.items():
        review = review.replace(contraction, replacement)

    # Remove non-alphanumeric and symbols
    review = re.sub(alphaPattern, ' ', review)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    review = re.sub(r'/', ' / ', review)
    return review
#loading tokenizer
def token():
    with open('tokenizer_new.pickle', 'rb') as handle:
      tokenizer = pickle.load(handle)
    return tokenizer

#preprocessing the sentence
def textclean(seentence):
    statement = [seentence]
    # print(statement)
    for i in range(len(statement)):
        print(i)
        listr = []
        statement[i] = preprocess_reviews(statement[i])
        for word in statement[i].split():
            print(word)
            if word.lower() not in stop_words:
                listr.append(word)
        statement[i] = " ".join(listr)
    # print(text)
    # print(statement)
    return statement
def model_pred(sentence):
    # print(tokenizer.texts_to_sequences(statement))
    tokenizer = token()
    review = textclean(sentence)
    review = pad_sequences(tokenizer.texts_to_sequences(review) , maxlen=input_length)
    review.shape
    pred = new_model.predict(review)
    print("LSTM Bidirectional :" , pred)
    result = ""
    score = pred[0][0]
    if score >= 0.70:
        result += "Positive"
    elif score <= 0.40:
        result += "Negative"
    else:
        result += "Neutral"
    # logging(data,result)
    print(result)
    return score,result
# model_res("Entertainment guranteed movie")