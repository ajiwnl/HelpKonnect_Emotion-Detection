import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load the Hugging Face model and tokenizer
model_name = "bhadresh-savani/bert-base-go-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with PyTorch weights
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)

# Save the model to a directory
model.save_pretrained("bert_go_emotion_modelv3")
tokenizer.save_pretrained("bert_go_emotion_modelv3")

# Define a function for the input signature (with correct input shape)
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 512], dtype=tf.int32)])
def model_signature(input_ids):
    return model(input_ids)

# Save the model to TensorFlow SavedModel format with input signature
tf.saved_model.save(model, "saved_model/bert_go_emotion_modelv3", signatures={"serving_default": model_signature})

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/bert_go_emotion_modelv3")

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model
with open("bert_go_emotion_modelv3.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved successfully.")
