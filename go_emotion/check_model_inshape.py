import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="bert_go_emotion_model.tflite")
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input shape
for detail in input_details:
    print("Input shape:", detail['shape'])
