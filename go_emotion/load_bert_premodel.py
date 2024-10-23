import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load a pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)  # 8 for your selected emotions

# Example input
input_text = "Today is a beautiful day, filled with sunshine and laughter, reminding me of all the wonderful moments life has to offer. I feel a burst of happiness as I watch my loved ones smiling and enjoying each other's company, creating memories that will last a lifetime. The sweet sound of my favorite song fills the air, igniting a wave of joy that makes me want to dance and celebrate life. With every little achievement I experience, I am reminded of the beauty of progress and the joy it brings to my heart."

# Preprocess input
inputs = tokenizer(input_text, return_tensors="tf", truncation=True, padding=True)

# Model inference
outputs = model(**inputs)
logits = outputs.logits

# Convert logits to probabilities
probabilities = tf.nn.softmax(logits, axis=-1)

# Print probabilities for each emotion
emotion_labels = ["Anxiety", "Envy", "Fear", "Joy", "Disgust", "Anger", "Embarrassment", "Sadness"]
for i, prob in enumerate(probabilities[0]):
    print(f"Predicted emotion - {emotion_labels[i]}: {prob.numpy():.2f}")