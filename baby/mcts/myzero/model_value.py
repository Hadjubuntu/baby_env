import tensorflow as tf

import tensorflow as tf
import tensorflow.keras.losses as kl
import tensorflow.keras.optimizers as ko

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv2D, Input, LSTM, Embedding, Dropout, Activation, Flatten
)
from tensorflow.keras.metrics import Mean


import numpy as np

# Eager execution needed
tf.enable_eager_execution()

def dense(input_shape, name=None):
    # Create inputs
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)

    # Create one dense layer and one layer for output
    x = Dense(128, activation='tanh')(x)
    dist = Dense(1)(x)

    # Finally build model
    model = Model(inputs=inputs, outputs=dist, name=name)
    model.summary()

    return model


class ModelValue:
    def __init__(self, input_shape, batch_size=32, lr=0.01):
        self.model = dense(input_shape, name="value_model")
        self.loss = kl.MeanSquaredError()
        self.optim = ko.Adam(lr=lr)
        # Loss evaluator
        self.eval_loss = Mean('loss')
        self.current_loss = 0.0

    def train(self, batch_data):
        x, y = [], []
        for data in batch_data:
            x.append(data['obs'])
            y.append(data['value'])

        x = np.array(x)
        y = np.array(y)
        loss_mse = self.fast_apply_gradients(x, y)

        return loss_mse

    @tf.function
    def predict(self, x):
        return self.model(x)

    @tf.function
    def fast_apply_gradients(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            loss = self.loss(y, predictions)

        trainable_vars = self.model.trainable_weights

        gradients = tape.gradient(loss, trainable_vars)
        self.optim.apply_gradients(zip(gradients, trainable_vars))
        self.current_loss = self.eval_loss(loss)

        return self.current_loss