import flask
import googleapiclient.discovery
from flask import Flask, request
from google.api_core.client_options import ClientOptions
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

app = Flask(__name__)

df = pd.read_csv('./static/train.csv.zip')
sentences = df['comment_text'].values
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences)
app.run(debug=True)

endpoint = 'https://us-central1-ml.googleapis.com'
client_options = ClientOptions(api_endpoint=endpoint)
service = googleapiclient.discovery.build('ml', 'v1', client_options=client_options)


@app.route('/')
def hello_world():
    return "hello, world"


@app.route('/predict/', methods=['POST'])
def predict():
    global tokenizer
    sentence = request.get_json()['sentence']
    sentence_array = np.array([sentence])
    tokenizer.fit_on_texts(sentence_array)
    sequence = tokenizer.texts_to_sequences(sentence_array)
    padded_sequence = pad_sequences(sequence, maxlen=100)
    predictions = predict_json("treehacks-304708", "toxicity_detection", padded_sequence.tolist(), "first_version")
    predictions_boolean = np.greater_equal(predictions, 0.5).tolist()
    response = {
        "predictions": predictions_boolean
    }
    return flask.jsonify(response)


def predict_json(project, model, instances, version=None):
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
