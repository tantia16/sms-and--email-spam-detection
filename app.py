from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and the vectorizer used for feature extraction
with open('spam_detector.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:  # Load the vectorizer
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    # Transform the input message using the same vectorizer used during training
    message_vectorized = vectorizer.transform([message]).toarray()  # Convert message to numeric features
    
    # Predict using the model
    prediction = model.predict(message_vectorized)[0]
    
    # Display the result based on the prediction
    result = "Ham" if prediction == 1 else "Spam"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
