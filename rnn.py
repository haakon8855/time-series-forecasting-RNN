"""haakon8855"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from data_loader import DataLoader


class RecurringNeuralNetwork:
    """
    RNN for pred
    """

    def __init__(self,
                 datapoint_width: int,
                 num_points: int,
                 weights_path: str = 'models/rnn'):
        self.datapoint_width = datapoint_width
        self.num_points = num_points
        self.weights_path = weights_path

        self.model = ks.Sequential()
        self.model.add(ks.layers.InputLayer((num_points, datapoint_width)))
        self.model.add(ks.layers.LSTM(64))
        self.model.add(ks.layers.Dense(8, 'relu'))
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
        self.model.fit(x=train_x,
                       y=train_y,
                       validation_data=validation_data,
                       batch_size=batch_size,
                       epochs=epochs)
        self.model.save_weights(filepath=self.weights_path)

    def predict(self, input_x):
        """
        Provide prediction based on the network input given.
        """
        return self.model(input_x)

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
            self.fit(train_x,
                     train_y,
                     validation_data,
                     batch_size=batch_size,
                     epochs=epochs)


def main():
    """
    Main method for rnn script.
    """
    weights_path = 'models/test10epochs50steps_test'
    steps = 50
    amount_to_remove = 288
    cols_to_use = [
        'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
        'flow', 'y_yesterday', 'y_prev', 'cos_minute', 'sin_minute',
        'cos_weekday', 'sin_weekday', 'cos_yearday', 'sin_yearday'
    ]

    data_loader = DataLoader()
    data_train = data_loader.get_processed_data('datasets\\no1_train.csv')
    data_valid = data_loader.get_processed_data('datasets\\no1_validation.csv')

    train_x, train_y = DataLoader.strip_and_format_data(
        data_train, cols_to_use, 'y', amount_to_remove, steps)
    valid_x, valid_y = DataLoader.strip_and_format_data(
        data_valid, cols_to_use, 'y', amount_to_remove, steps)

    network = RecurringNeuralNetwork(len(cols_to_use),
                                     num_points=steps,
                                     weights_path=weights_path)
    network.fit_or_load_model(train_x,
                              train_y, (valid_x, valid_y),
                              batch_size=32,
                              epochs=10)

    limit = 200
    test_x = valid_x[:limit]
    test_y = valid_y[:limit]
    y_pred = network.predict(test_x)
    print(f"pred: {y_pred[0][0]}, corr: {train_y[0]}")

    plt.figure(figsize=(15, 7))
    plt.title('Prediction')
    plt.plot(test_y, label='target')
    plt.plot(y_pred, label='pred')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
