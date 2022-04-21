"""haakon8855"""

from msilib.schema import TextStyle
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
                           optimizer=ks.optimizers.Adam(learning_rate=0.0001),
                           metrics=[ks.losses.MeanSquaredError()])

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
        Assume data_x contains 12 cases.
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

    @staticmethod
    def format_input_data(data, steps):
        """
        Returns the data on the correct format.
        """
        formatted_data = []
        dataframe = data.copy()
        for i in range(len(dataframe.index) - steps + 1):
            formatted_data.append(dataframe.iloc[i:i + steps])
        return np.array(formatted_data)

    @staticmethod
    def format_target_data(target, steps):
        """
        Returns the data on the correct format.
        """
        return np.array(target[steps - 1:])


def main():
    """
    Main method for rnn script.
    """
    data_loader_train = DataLoader('datasets\\no1_train.csv')
    data_loader_valid = DataLoader('datasets\\no1_validation.csv')
    data_train = data_loader_train.get_data()
    data_valid = data_loader_valid.get_data()
    cols_to_use = [
        'hydro',
        'micro',
        'thermal',
        'wind',
        'river',
        'total',
        'sys_reg',
        'flow',
        'y_yesterday',
        'y_prev',
        'cos_minute',
        'sin_minute',
        'cos_weekday',
        'sin_weekday',
        'cos_yearday',
        'sin_yearday',
    ]

    steps = 12
    network = RecurringNeuralNetwork(
        len(cols_to_use),
        num_points=steps,
        weights_path='models/test10epochswithvalid')

    idx_to_remove = 288
    data_x_train_stripped = data_train.iloc[idx_to_remove:][cols_to_use]
    data_y_train_stripped = data_train.iloc[idx_to_remove:]['y']
    train_x = RecurringNeuralNetwork.format_input_data(data_x_train_stripped,
                                                       steps)
    train_y = RecurringNeuralNetwork.format_target_data(
        data_y_train_stripped, steps)
    data_x_valid_stripped = data_valid.iloc[idx_to_remove:][cols_to_use]
    data_y_valid_stripped = data_valid.iloc[idx_to_remove:]['y']
    valid_x = RecurringNeuralNetwork.format_input_data(data_x_valid_stripped,
                                                       steps)
    valid_y = RecurringNeuralNetwork.format_target_data(
        data_y_valid_stripped, steps)

    loaded = network.load_all_weights()
    if not loaded:
        network.fit(train_x,
                    train_y, (valid_x, valid_y),
                    batch_size=32,
                    epochs=10)
    limit = 200
    test_x = valid_x[:limit]
    test_y = valid_y[:limit]
    # test_x = train_x[:limit]
    # test_y = train_y[:limit]
    y_pred = network.predict(test_x)
    print(f"pred: {y_pred[0][0]}, corr: {train_y[0]}")

    plt.figure(figsize=(15, 7))
    plt.title('Prediction')
    plt.plot(test_y)
    plt.plot(y_pred)
    plt.show()


if __name__ == '__main__':
    main()
