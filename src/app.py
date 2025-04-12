from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
with open('fake_news_model.pkl', 'rb') as file:
    artifacts = pickle.load(file)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']

# Preprocessing function (same as in training)
def preprocess(text):
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords

    nltk.download('stopwords')
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = str(text).lower()
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]

    return ' '.join(filtered_words)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_text = data.get('text', '')

        if not raw_text:
            return jsonify({'error': 'Missing text in request'}), 400

        # Preprocess the input text
        processed_text = preprocess(raw_text)

        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])

        # Make prediction
        prediction = model.predict(text_vector)[0]
        confidence = model.predict_proba(text_vector)[0].max()

        # Return prediction result
        result = "Fake News" if prediction == 1 else "Real News"
        return jsonify({
            'prediction': result,
            'confidence': float(confidence)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "Fake News Detection API"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
