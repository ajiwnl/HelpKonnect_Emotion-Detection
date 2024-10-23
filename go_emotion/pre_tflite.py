import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

# Load the tokenizer
model_name = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="bert_go_emotion_modelv3.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input details for debugging
print("Input Details:", input_details)
print("Output Details:", output_details)

# Function to preprocess input text
def preprocess_input(text):
    # Tokenize and encode the input text
    inputs = tokenizer(
        text,
        return_tensors='np',
        max_length=512,  # Use the appropriate max length
        padding='max_length',  # Pad to max length
        truncation=True
    )
    return inputs['input_ids'], inputs['attention_mask']  # Return input_ids and attention_mask

# Input from user
user_input = input("Enter a journal entry for emotion analysis: ")

# Preprocess the input
input_ids, attention_mask = preprocess_input(user_input)

# Convert inputs to np.int32
input_ids = np.array(input_ids, dtype=np.int32)
attention_mask = np.array(attention_mask, dtype=np.int32)

# Ensure input shapes match model expectations (should be [1, 512])
input_ids = input_ids.reshape([1, 512])
attention_mask = attention_mask.reshape([1, 512])

# Check input_details to determine correct indices
for i, detail in enumerate(input_details):
    print(f"Input {i}: Name: {detail['name']}, Index: {detail['index']}, Shape: {detail['shape']}")

# Set the input tensors (check input details for exact indices)
interpreter.set_tensor(input_details[0]['index'], input_ids)  # input_ids
# Check if attention_mask is available
if len(input_details) > 1:
    interpreter.set_tensor(input_details[1]['index'], attention_mask)  # attention_mask
else:
    print("No attention_mask tensor found.")

# Run inference
interpreter.invoke()

# Get the output from the model
output_data = interpreter.get_tensor(output_details[0]['index'])

# Output is a probability distribution over 28 emotion classes
print("Predicted emotion probabilities:", output_data)

# Apply softmax to get normalized probabilities
emotion_probs = tf.nn.softmax(output_data[0]).numpy()

# Convert probabilities to percentages
emotion_probs_percentage = emotion_probs * 100

# Emotion labels
emotion_labels = {
    "0": "admiration",
    "1": "amusement",
    "2": "anger",
    "3": "annoyance",
    "4": "approval",
    "5": "caring",
    "6": "confusion",
    "7": "curiosity",
    "8": "desire",
    "9": "disappointment",
    "10": "disapproval",
    "11": "disgust",
    "12": "embarrassment",
    "13": "excitement",
    "14": "fear",
    "15": "gratitude",
    "16": "grief",
    "17": "joy",
    "18": "love",
    "19": "nervousness",
    "20": "optimism",
    "21": "pride",
    "22": "realization",
    "23": "relief",
    "24": "remorse",
    "25": "sadness",
    "26": "surprise",
    "27": "neutral"
}

# Print all normalized probabilities with their corresponding emotion labels
for index, prob in enumerate(emotion_probs_percentage):
    emotion_label = emotion_labels[str(index)]
    print(f"{emotion_label}: {prob:.2f}%")  # Display as percentage

# Find the emotion with the highest probability
predicted_emotion = np.argmax(emotion_probs)
predicted_emotion_label = emotion_labels[str(predicted_emotion)]
print(f"\nPredicted emotion class: {predicted_emotion}")
print(f"Predicted emotion label: {predicted_emotion_label}")

# Find the four highest probabilities
top_indices = np.argsort(emotion_probs)[-4:][::-1]  # Get the top 4 indices
top_probabilities = emotion_probs_percentage[top_indices]

print("\nTop 4 predicted emotions:")
for idx in range(len(top_indices)):
    emotion_label = emotion_labels[str(top_indices[idx])]
    print(f"{emotion_label}: {top_probabilities[idx]:.2f}%")  # Display as percentage