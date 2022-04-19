"""haakon8855"""

import numpy as np
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

    def fit(self, data_train, cols_to_use, epochs=5):
        """
        Trains the network on the given dataset.
        """
        formatted_x = self.format_input_data(data_train, cols_to_use)
        formatted_y = self.format_target_data(data_train['y'])
        # print(f"lenx:{len(formatted_x)}")
        # print(f"leny:{len(formatted_y)}")

        self.model.fit(x=formatted_x, y=formatted_y, epochs=epochs)
        self.model.save_weights(filepath=self.weights_path)

    def format_input_data(self, data, cols_to_use):
        """
        Returns the data on the correct format.
        """
        formatted_data = []
        dataframe = data.iloc[288:][cols_to_use]
        for i in range(len(dataframe.index) - self.num_points):
            formatted_data.append(dataframe.iloc[i:i + self.num_points])
        return np.array(formatted_data)

    def format_target_data(self, target):
        """
        Returns the data on the correct format.
        """
        target = target[288:]
        return np.array(target[self.num_points:])

    def load_all_weights(self):
        """
        Load weights
        """
        try:
            self.model.load_weights(filepath=self.weights_path)
        except ValueError:
            print(
                "Unable to read weights from file. Starts training from scratch."
            )


def main():
    """
    Main method for rnn script.
    """
    data_loader = DataLoader('datasets\\no1_train.csv')
    data_train = data_loader.get_data()
    cols_to_use = [
        'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
        'flow', 'y_yesterday', 'y_prev', 'cos_minute', 'sin_minute',
        'cos_weekday', 'sin_weekday', 'cos_yearday', 'sin_yearday'
    ]
    steps = 12
    network = RecurringNeuralNetwork(len(cols_to_use),
                                     num_points=steps,
                                     weights_path='models/test')
    network.fit(data_train, cols_to_use, epochs=1)


if __name__ == "__main__":
    main()
