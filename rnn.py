"""haakon8855"""

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
        self.input_shape = (datapoint_width, num_points)
        self.weights_path = weights_path

        self.model = ks.Sequential()
        self.model.add(ks.layers.InputLayer(self.input_shape))
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
        # self.model.fit()
        # self.model.save_weights(filepath=self.weights_path)

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
    network.fit(data_train, cols_to_use, epochs=4)


if __name__ == "__main__":
    main()
