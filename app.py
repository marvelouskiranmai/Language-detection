from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Load the preprocessed data
preprocessed_path = 'preprocessed.csv'
df = pd.read_csv(preprocessed_path)

# TF-IDF representation
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['Text'])

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X, df['Language'])

# Save the trained model and vectorizer
model_path = 'language_detection_model.joblib'
vectorizer_path = 'tfidf_vectorizer.joblib'
joblib.dump(svm_model, model_path)
joblib.dump(tfidf_vectorizer, vectorizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input text from the request
        data = request.get_json()
        text = data['text']

        # Preprocess and vectorize the input text
        text = text.lower()
        text_tfidf = tfidf_vectorizer.transform([text])

        # Predict the language
        prediction = svm_model.predict(text_tfidf)[0]

        # Return the prediction
        return jsonify({'language': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
