import os

if os.path.exists("bert_go_emotion_model.tflite"):
    print("TFLite model file exists.")
else:
    print("TFLite model file does not exist.")