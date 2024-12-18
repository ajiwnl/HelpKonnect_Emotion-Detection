from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the Hugging Face model and tokenizer
model_name = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)

# Define emotion labels (for 28 emotion classes)
emotion_labels = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring",
    "Confusion", "Curiosity", "Desire", "Disappointment", "Disapproval",
    "Disgust", "Embarrassment", "Excitement", "Fear", "Gratitude",
    "Grief", "Joy", "Love", "Nervousness", "Optimism", "Pride",
    "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
]

# Function to process the input and make a prediction
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    logits = model(inputs["input_ids"]).logits
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Convert probabilities to percentages
    probabilities_percentage = probabilities.numpy() * 100

    # Find the predicted class
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]

    return probabilities_percentage[0], predicted_class  # Return probabilities and predicted class


# Define a route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Expect JSON data for POST requests
        if request.is_json:
            data = request.get_json()
            if 'text' not in data:
                return jsonify({"error": "No text provided"}), 400
            text = data['text']
        else:
            return jsonify({"error": "Content-Type must be application/json"}), 415
    elif request.method == 'GET':
        # For GET requests, read 'text' as a query parameter
        text = request.args.get('text', None)
        if not text:
            return jsonify({"error": "No text provided"}), 400

    probabilities, predicted_class = predict_emotion(text)

    # Prepare the response
    response = {
        "predicted_emotion": emotion_labels[predicted_class],
        "probabilities": {emotion_labels[i]: float(prob) for i, prob in enumerate(probabilities)},
        "top_4_emotions": []
    }

    # Get the top 4 emotions
    top_indices = np.argsort(probabilities)[-4:][::-1]
    top_probabilities = probabilities[top_indices]

    for idx in range(len(top_indices)):
        response["top_4_emotions"].append({
            "emotion": emotion_labels[top_indices[idx]],
            "probability": round(float(top_probabilities[idx]), 2)
        })

    return jsonify(response)

if __name__ == '__main__':
    #app.run()
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)