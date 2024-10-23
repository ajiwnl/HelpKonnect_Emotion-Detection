# BERT-based Emotion Classifier API

This is a Flask-based API that uses the `bert-base-go-emotion` model from Hugging Face's Transformers library to classify text into 28 different emotion categories. It provides an endpoint to predict the emotions conveyed in a given text input.

## Features

- Uses a pre-trained BERT model to predict emotions.
- Supports 28 distinct emotion labels such as joy, sadness, anger, and more.
- Returns the predicted emotion along with the probabilities for all emotion classes.
- Additionally, returns the top 4 emotions predicted with their corresponding probabilities.

## Emotion Labels

The model classifies text into the following 28 emotions: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## Requirements

To run this project, you need the following libraries, which are specified in the `requirements.txt`.

