"""Haakon8855"""

import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from configuration import Config
from data_loader import DataLoader
from rnn import RecurringNeuralNetwork


class ImbalanceForecasting:
    """
    Runs a recurring neural network to predict the future imbalance in power
    grids.
    """

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = Config.get_config(config_file)

        # Fetch config
        global_conf = self.config['GLOBALS']
        self.model_name = global_conf['model_name']
        self.weights_path = f'models/{self.model_name}'
        self.steps = int(global_conf['steps'])
        self.max_future_steps = int(global_conf['max_future_steps'])
        self.pred_start = int(global_conf['pred_start'])
        self.hist_size = int(global_conf['hist_size'])
        self.amount_to_remove = int(global_conf['amount_to_remove'])
        self.randomize_y_prev = global_conf['randomize_y_prev'] == 'True'
        self.randomize_plot_location = global_conf[
            'randomize_plot_location'] == 'True'
        self.cols_to_use = json.loads(global_conf['cols_to_use'])

        self.data_train = None
        self.data_valid = None
        self.train_x = None
        self.train_y = None
        self.valid_x = None
        self.valid_y = None
        self.history = None

        # Initialize classes
        self.data_loader = DataLoader()
        self.network = RecurringNeuralNetwork(len(self.cols_to_use),
                                              steps=self.steps,
                                              weights_path=self.weights_path)

    def load_data(self):
        """
        Loads the training and verification data from file and processes it.
        """
        self.data_train = self.data_loader.get_processed_data(
            'datasets\\no1_train.csv')
        self.data_valid = self.data_loader.get_processed_data(
            'datasets\\no1_validation.csv')

        self.train_x, self.train_y = DataLoader.strip_and_format_data(
            self.data_train,
            self.cols_to_use,
            'y',
            self.amount_to_remove,
            self.steps,
            randomize_y_prev=self.randomize_y_prev)
        self.valid_x, self.valid_y = DataLoader.strip_and_format_data(
            self.data_valid,
            self.cols_to_use,
            'y',
            self.amount_to_remove,
            self.steps,
            randomize_y_prev=self.randomize_y_prev)

    def train(self):
        """
        Trains the network on the dataset or loads pretrained weights if
        available.
        """
        self.history = self.network.fit_or_load_model(
            self.train_x,
            self.train_y, (self.valid_x, self.valid_y),
            batch_size=32,
            epochs=10)

    def plot_loss(self):
        """
        Plots the training and validation loss during training.
        """
        if self.history is not None:
            plot_loss_history(self.history, self.model_name)

    def predict_one_step(self, num_plots: int = 2, limit: int = 300):
        """
        Predicts one step ahead and plots the resulting imbalance graph along
        with the target.
        """
        for i in range(num_plots):
            plot_future_one_step(self.pred_start + i * limit, limit,
                                 self.valid_x, self.valid_y, self.network)

    def predict_two_hours(self,
                          num_plots: int = 2,
                          num_plots_per_plot: int = 9):
        """
        Predicts two hours into the future by using the predicted values
        as y_prev for continued predictions.
        """
        for _ in range(num_plots):
            plot_pred_future_multiple(
                self.data_valid,
                self.network,
                num_plots_per_plot,
                self.hist_size,
                self.amount_to_remove,
                self.pred_start,
                self.steps,
                self.max_future_steps,
                self.cols_to_use,
                randomize_location=self.randomize_plot_location)

    def run(self):
        """
        Trains on the training data or loads weights,
        and then forecasts the imbalance.
        """
        self.load_data()
        self.train()
        self.plot_loss()
        self.predict_one_step()
        self.predict_two_hours()


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
    Main function for running this python script.
    """
    imbalance_forecasting = ImbalanceForecasting(
        "config/config1.ini")  # demo config, free to edit
    imbalance_forecasting.run()


if __name__ == "__main__":
    main()
