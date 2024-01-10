import streamlit as st
from keras import backend as K
from predict import Load_model, token, textclean
from tensorflow.keras.preprocessing.sequence import pad_sequences

def simpleui():
    st.title('Sentimental analysis for movie review')
    st.write('A simple sentiment analysis classification app')
    st.subheader('Input movie review below')
    sentence = st.text_area('Enter your review here',height=200)
    predict_btt = st.button('predict')
    model, session = Load_model()
    tokenizer = token()
    review = textclean(sentence)
    data = pad_sequences(tokenizer.texts_to_sequences(review) , maxlen=750)
    prediction = model.predict(data)
    score = prediction[0][0]
    if score >= 0.70:
        st.success('Positive review')
    elif score <= 0.40:
        st.success('Negative Review')
    else:
        st.success('Neutral Review')