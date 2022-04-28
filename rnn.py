"""haakon8855"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks

from data_loader import DataLoader


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


def plot_future_one_step(pred_start: int, limit: int, valid_x: np.array,
                         valid_y: np.array, network: RecurringNeuralNetwork):
    """
    Predicts one step into the future for 'limit' amount of timesteps.
    """
    test_x = valid_x[pred_start:pred_start + limit]
    test_y = valid_y[pred_start:pred_start + limit]
    y_pred = network.predict(test_x)
    x_axis = np.arange(pred_start, pred_start + limit)

    plt.figure(figsize=(15, 7))
    plt.title('Predict one step ahead')
    plt.plot(x_axis, test_y, label='target')
    plt.plot(x_axis, y_pred, label='pred')
    plt.legend()
    plt.show()


def get_future_plots(data_valid: pd.DataFrame, network: RecurringNeuralNetwork,
                     hist_size: int, amount_to_remove: int, pred_start: int,
                     steps: int, max_future_steps: int, cols_to_use: list):
    """
    Predicts 'steps' amount of steps into the future. Using the previously
    predicted value as input for the next prediction.
    """
    preds_idx_start = amount_to_remove + pred_start
    preds_idx_stop = amount_to_remove + pred_start + steps + max_future_steps
    pred_input = data_valid.iloc[preds_idx_start:preds_idx_stop][cols_to_use]
    preds = network.predict_into_future(pred_input)

    target_idx_start = amount_to_remove + pred_start - hist_size
    target_idx_stop = preds_idx_stop
    target = data_valid.iloc[target_idx_start:target_idx_stop]['y']

    hist = list(target[:hist_size + steps])
    last_hist = hist[-1]
    target = target[hist_size + steps:hist_size + steps + max_future_steps]
    target = list(target)
    target.insert(0, last_hist)
    preds = list(preds)
    preds.insert(0, last_hist)

    historic = (np.arange(0, hist_size + steps) + pred_start, hist)
    targets = (np.arange(hist_size + steps - 1,
                         hist_size + steps + max_future_steps) + pred_start,
               target)
    predictions = (np.arange(hist_size + steps - 1, hist_size + steps +
                             max_future_steps) + pred_start, preds)
    return historic, targets, predictions


def plot_pred_future_multiple(data_valid: pd.DataFrame,
                              network: RecurringNeuralNetwork,
                              num_plots: int,
                              hist_size: int,
                              amount_to_remove: int,
                              pred_start: int,
                              steps: int,
                              max_future_steps: int,
                              cols_to_use: list,
                              randomize_location: bool = True):
    """
    Predicts 'steps' amount of steps into the future. Using the previously
    predicted value as input for the next prediction.
    """
    rows = 3
    fig, axs = plt.subplots(nrows=rows,
                            ncols=(int(np.ceil(num_plots / rows))),
                            figsize=(15, 10))
    for i in range(len(axs.ravel())):
        axi = axs.ravel()[i]
        start_location = pred_start + i * max_future_steps
        if randomize_location:
            start_location = np.random.randint(
                amount_to_remove * 2,
                len(data_valid) - amount_to_remove)
        historic, targets, predictions = get_future_plots(
            data_valid, network, hist_size, amount_to_remove, start_location,
            steps, max_future_steps, cols_to_use)
        axi.set_title(f'Pred #{i}')
        axi.plot(historic[0], historic[1], label='hist')
        axi.plot(predictions[0], predictions[1], label='pred')
        axi.plot(targets[0], targets[1], label='target')
    fig.suptitle('Predict 2 hrs')
    fig.legend(['hist', 'pred', 'target'])
    plt.show()


def plot_loss_history(history, model_name: str = "loss_plot"):
    """
    Plots the loss over time during training.
    """
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.clf()
    plt.title('Loss')
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.legend()
    plt.savefig(f'plots/{model_name}.png')
    plt.show()


def main():
    """
    Main method for rnn script.
    """
    # model_name = '8epochs_144steps_gru_25drop_goodfit'
    # model_name = '20epochs_144steps_gru_25drop_goodfit'
    model_name = '10epochs_144steps_gru_25drop_goodfit'
    weights_path = f'models/{model_name}'
    steps = 144
    max_future_steps = 24
    pred_start = 4000
    hist_size = 0  #min(pred_start, steps)
    amount_to_remove = 288
    cols_to_use = [
        'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
        'flow', 'y_yesterday', 'y_prev', 'cos_minute', 'sin_minute',
        'cos_weekday', 'sin_weekday', 'cos_yearday', 'sin_yearday'
    ]
    print("Frick")

    data_loader = DataLoader()
    data_train = data_loader.get_processed_data('datasets\\no1_train.csv')
    data_valid = data_loader.get_processed_data('datasets\\no1_validation.csv')
    print("Frick2")

    train_x, train_y = DataLoader.strip_and_format_data(
        data_train, cols_to_use, 'y', amount_to_remove, steps)
    valid_x, valid_y = DataLoader.strip_and_format_data(
        data_valid, cols_to_use, 'y', amount_to_remove, steps)
    print("Frick3")

    network = RecurringNeuralNetwork(len(cols_to_use),
                                     steps=steps,
                                     weights_path=weights_path)
    print("Frick4")
    history = network.fit_or_load_model(train_x,
                                        train_y, (valid_x, valid_y),
                                        batch_size=32,
                                        epochs=10)
    print("Frick5")

    if history is not None:
        plot_loss_history(history, model_name)

    limit = 300
    for i in range(2):
        plot_future_one_step(pred_start + i * limit, limit, valid_x, valid_y,
                             network)

    num_plots = 9
    for _ in range(2):
        plot_pred_future_multiple(data_valid,
                                  network,
                                  num_plots,
                                  hist_size,
                                  amount_to_remove,
                                  pred_start,
                                  steps,
                                  max_future_steps,
                                  cols_to_use,
                                  randomize_location=True)


if __name__ == '__main__':
    main()
