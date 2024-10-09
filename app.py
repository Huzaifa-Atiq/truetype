from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('textdetector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    probabilities = model.predict_proba(vectorized_text)

    return jsonify({
        'prediction': prediction[0],
        'probabilities': probabilities[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
