#! /usr/bin/env python3
# coding: utf-8

import os
import joblib
from flask import Flask, request, jsonify
import pathlib
import html
import preprocessor as tweet_preprocessor

import spacy

# import cv2
# import tflite_runtime.interpreter as tflite

import tensorflow as tf
from tensorflow import keras
# import numpy as np

try:
    from tensorflow.keras.layers import TextVectorization
except ImportError:
    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

app = Flask(__name__)

# --- Load Spacy ---
print("Load spaCy lemmatizer")
nlp = spacy.load('en_core_web_sm')


def tokenize(text):

    # tokenisation
    tokens = nlp(text)
    print("tokens:", tokens)

    # lemmatize
    lemmas = [x.lemma_ for x in tokens]
    print("lemmas:", lemmas)
    return " ".join(lemmas)


def preprocess_txt(string):
    print("PRE01:", string)

    # suppression des majuscules
    text = string.lower()

    # suppression des espaces au début et à la fin des textes
    text = text.strip()
    print("PRE02:", text)

    text = html.unescape(text)
    print("PRE03:", text)

    tweet_preprocessor.set_options(tweet_preprocessor.OPT.MENTION, tweet_preprocessor.OPT.RESERVED, tweet_preprocessor.OPT.EMOJI)
    text = tweet_preprocessor.clean(text)
    print("PRE04:", text)

    tweet_preprocessor.set_options(tweet_preprocessor.OPT.URL)
    text = tweet_preprocessor.tokenize(text)
    print("PRE05:", text)

    lemmas = tokenize(text)
    print("PRE06:", lemmas)

    return lemmas


# --- Load TextVectorizer ---
print("Load TextVectorizer Model")
TV_config, TV_weigths = joblib.load(pathlib.Path("models", "SelectedTextVectorizerModel.bin"))
text_vectorization = TextVectorization.from_config(TV_config)
# You have to call `adapt` with some dummy data (BUG in Keras)
# text_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
text_vectorization.set_weights(TV_weigths)

# --- Load TF Model ---
print("Load Classification Model")
model = keras.models.load_model("models/SelectedModel.keras")

# --- Load TF-Lite model using an interpreter
# interpreter = tflite.Interpreter(model_path="models/model1extra.tflite")
# interpreter.allocate_tensors()
# input_index = interpreter.get_input_details()[0]["index"]
# output_index = interpreter.get_output_details()[0]["index"]


@app.route("/")
def index():
    return "Hello world !<br>The 'Twitter Sentiment Analysis API' server is up."


@app.route("/predict", methods=["POST"])
def predict():

    try:
        print("--- Collect >>>> ", request.data)
        raw_txt = str(request.data)

    except Exception as e:
        print("Error:", e)

    # Preprocess string
    txt = preprocess_txt(raw_txt)

    # Apply TextVectorizer
    print("--- TextVectorization")
    txt = text_vectorization(txt)

    # Convert to Tensor
    print("--- Convert to tensor")
    ready_txt = tf.convert_to_tensor([txt])

    # Apply model
    print("--- Predict")
    pred = model.predict(ready_txt)

    if pred[0] > 0.474:  # best threshold found
        label = "Positive"
        pred_value = float(pred[0])
    else:
        label = "Negative"
        pred_value = 1.0 - float(pred[0])

    # Return values
    return jsonify(
            f"The predicted label is **{label.upper()}** with the following probability: {pred_value*100.0:.2f}%"
    )


print("Server ready")

if __name__ == "__main__":
    # print("ICI on y va ")
    # raw_txt = tf.convert_to_tensor(["I hate it", "I love it"])
    # raw_txt = text_vectorization(raw_txt)
    # print(model.predict(raw_txt))

    current_port = int(os.environ.get("PORT") or 5000)
    app.run(debug=True, host="0.0.0.0", port=current_port)
