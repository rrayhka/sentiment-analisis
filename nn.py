import os
import pandas as pd
import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Load dataset
data = pd.read_csv('dataset/dataset.csv')
data = data[["normalized", "sentiment"]]
data["label"] = data["sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})

# Tokenizer
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(data["normalized"])

def prepocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = keras.utils.pad_sequences(sequences, maxlen=512, padding='post', truncating='post')
    return padded

def create_bilstm_model(vocab_size, output_size):
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=[1]),
            keras.layers.Embedding(
                vocab_size, 512, embeddings_regularizer=keras.regularizers.l2(1e-6)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    64,
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                )
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    128,
                    return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                )
            ),
            keras.layers.Bidirectional(
                keras.layers.LSTM(256, kernel_regularizer=keras.regularizers.l2(1e-6))
            ),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(output_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=True,
    )

    return model

def create_attention_model(vocab_size, output_size):
    # Input layer
    input = keras.layers.Input(shape=(512,))
    embeddings = keras.layers.Embedding(
        vocab_size, 512, embeddings_regularizer=keras.regularizers.l2(1e-6)
    )(input)
    dropout = keras.layers.Dropout(0.2)(embeddings)

    # Bi-LSTM
    lstm = keras.layers.Bidirectional(
        keras.layers.LSTM(
            64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-6)
        )
    )(dropout)

    # Attention
    norm = keras.layers.LayerNormalization()(lstm)
    attention = keras.layers.Attention(dropout=0.1)([norm, norm])

    for _ in range(2):
        norm = keras.layers.LayerNormalization()(attention)
        attention = keras.layers.Attention(dropout=0.1)([norm, norm])

    # Fully connected
    fc = keras.layers.Flatten()(attention)
    fc = keras.layers.Dropout(0.2)(fc)
    output = keras.layers.Dense(output_size, activation="softmax")(fc)

    model = keras.models.Model(inputs=input, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5, weight_decay=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=True,
    )

    return model

bilstm_model = create_bilstm_model(20000, 3)
attention_model = create_attention_model(20000, 3)

bilstm_model.load_weights('models/bilstm.weights.h5')
attention_model.load_weights('models/attention.weights.h5')
def predict_sentiment(text, model):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = keras.utils.pad_sequences(sequences, maxlen=512, padding='post', truncating='post')
    prediction = model.predict(padded_sequences)
    label = tf.argmax(prediction, axis=1).numpy()[0]
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[label]

user_input = "saya suka presiden yang korupsi"
bilstm_sentiment = predict_sentiment(user_input, bilstm_model)
attention_sentiment = predict_sentiment(user_input, attention_model)

# print(f"Bi-LSTM model prediction: {bilstm_sentiment}")
# print(f"Attention model prediction: {attention_sentiment}")