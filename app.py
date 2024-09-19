import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the trained model and vectorizer
with open(r'D:\Data Science Projects life\Fake News Predictor\Model Traning\Model (1).pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open(r'D:\Data Science Projects life\Fake News Predictor\Model Traning\vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def preprocess_text(text):
    # Preprocess the text (remove non-alphabetical characters, stopwords, stemming)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    text = [ps.stem(word) for word in text if word not in set(all_stopwords)]
    return ' '.join(text)

def predict_note_authentication(text):
    # Preprocess the text
    text = preprocess_text(text)

    # Transform the text using the loaded vectorizer (using transform, not fit_transform)
    text_transformed = vectorizer.transform([text]).toarray()

    # Make the prediction
    return classifier.predict(text_transformed)

def main():
    st.title("Fake News Detector")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Fake News Detector</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Text = st.text_area("Enter Text", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(Text)
        if result[0] == 0:
            result = "This News is Real."
        else:
            result = "This News is Fake."    
    st.success(result)


if __name__ == '__main__':
    main()
