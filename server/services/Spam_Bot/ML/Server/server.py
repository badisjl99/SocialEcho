from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Determine the absolute path to the ModelOutput folder
model_output_path = os.path.join(os.path.dirname(__file__), '..', 'ModelOutput')

# Load the model and vectorizer
model = joblib.load(os.path.join(model_output_path, 'spam_classifier.pkl'))
vectorizer = joblib.load(os.path.join(model_output_path, 'vectorizer.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message')
    
    message_vector = vectorizer.transform([message])
    
    prediction = model.predict(message_vector)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
