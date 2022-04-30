"""haakon8855"""

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks


class RecurringNeuralNetwork:
    """
    RNN for pred
    """

    def __init__(self,
                 datapoint_width: int,
                 steps: int,
                 weights_path: str = 'models/rnn'):
        self.datapoint_width = datapoint_width
        self.steps = steps
        self.weights_path = weights_path

        self.model = ks.Sequential()
        self.model.add(ks.layers.InputLayer((steps, datapoint_width)))
        self.model.add(ks.layers.GRU(128, return_sequences=True))
        self.model.add(ks.layers.Dropout(0.25))
        self.model.add(ks.layers.GRU(64))
        self.model.add(ks.layers.Dropout(0.25))
        self.model.add(ks.layers.Dense(32, 'relu'))
        self.model.add(ks.layers.Dense(1, 'linear'))

        self.model.summary()

        self.initialize_model()

    def initialize_model(self):
        """
        Compiles the model.
        """
        self.model.compile(loss=ks.losses.MeanSquaredError(),
                           optimizer=ks.optimizers.Adam(learning_rate=0.0001))

    def fit(self,
            train_x,
            train_y,
            validation_data=None,
            batch_size=1024,
            epochs=5):
        """
        Trains the network on the given dataset.
        """
        history = self.model.fit(x=train_x,
                                 y=train_y,
                                 validation_data=validation_data,
                                 batch_size=batch_size,
                                 epochs=epochs)
        self.model.save_weights(filepath=self.weights_path)
        return history

    def predict(self, input_x):
        """
        Provide prediction based on the network input given.
        """
        return self.model(input_x)

    def predict_into_future(self, data, future_steps: int = 24):
        """
        Predicts the target value several timesteps into the future.
        """
        preds = []
        pred_i = data.iloc[self.steps - 1]['y_prev']
        for i in range(future_steps):
            data.iat[i + self.steps - 1,
                     data.columns.get_loc('y_prev')] = pred_i
            input_i = data.iloc[i:i + self.steps]
            input_i = np.array(input_i)[np.newaxis, :, :]
            pred_i = self.predict(input_i).numpy()[0][0]
            preds.append(pred_i)
        return np.array(preds)

    def load_all_weights(self):
        """
        Load weights
        """
        try:
            self.model.load_weights(filepath=self.weights_path)
            print('Loaded weights successfully')
            return True
        except tf.errors.NotFoundError:
            print(
                'Unable to read weights from file. Starts training from scratch.'
            )
        return False

    def fit_or_load_model(self,
                          train_x,
                          train_y,
                          validation_data,
                          batch_size=32,
                          epochs=10):
        """
        Tries to load weights from file. If no suitable weights are present,
        the model is retrained from scratch.
        """
        loaded = self.load_all_weights()
        if not loaded:
            history = self.fit(train_x,
                               train_y,
                               validation_data,
                               batch_size=batch_size,
                               epochs=epochs)
            return history
        return None
