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
        self.epochs = int(global_conf['epochs'])
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
            epochs=self.epochs)

    def plot_loss(self):
        """
        Plots the training and validation loss during training.
        """
        if self.history is not None:
            self.plot_loss_history(self.model_name)

    def predict_one_step(self, num_plots: int = 2, limit: int = 300):
        """
        Predicts one step ahead and plots the resulting imbalance graph along
        with the target.
        """
        for i in range(num_plots):
            self.plot_future_one_step(self.pred_start + i * limit, limit)

    def predict_two_hours(self,
                          num_plots: int = 2,
                          num_plots_per_plot: int = 9):
        """
        Predicts two hours into the future by using the predicted values
        as y_prev for continued predictions.
        """
        for _ in range(num_plots):
            self.plot_pred_future_multiple(num_plots_per_plot)

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

    def plot_future_one_step(self, pred_start: int, limit: int):
        """
        Predicts one step into the future for 'limit' amount of timesteps.
        """
        test_x = self.valid_x[pred_start:pred_start + limit]
        test_y = self.valid_y[pred_start:pred_start + limit]
        y_pred = self.network.predict(test_x)
        x_axis = np.arange(pred_start, pred_start + limit)

        plt.figure(figsize=(15, 7))
        plt.title('Predict one step ahead')
        plt.plot(x_axis, test_y, label='target')
        plt.plot(x_axis, y_pred, label='pred')
        plt.legend()
        plt.show()

    def get_future_plots(self, start_location: int):
        """
        Predicts 'steps' amount of steps into the future. Using the previously
        predicted value as input for the next prediction.
        """
        preds_idx_start = self.amount_to_remove + start_location
        preds_idx_stop = self.amount_to_remove + start_location + self.steps + self.max_future_steps
        pred_input = self.data_valid.iloc[preds_idx_start:preds_idx_stop][
            self.cols_to_use]
        preds = self.network.predict_into_future(pred_input)

        target_idx_start = self.amount_to_remove + start_location - self.hist_size
        target_idx_stop = preds_idx_stop
        target = self.data_valid.iloc[target_idx_start:target_idx_stop]['y']

        hist = list(target[:self.hist_size + self.steps])
        last_hist = hist[-1]
        target = target[self.hist_size + self.steps:self.hist_size +
                        self.steps + self.max_future_steps]
        target = list(target)
        target.insert(0, last_hist)
        preds = list(preds)
        preds.insert(0, last_hist)

        historic = (np.arange(0, self.hist_size + self.steps) + start_location,
                    hist)
        targets = (
            np.arange(self.hist_size + self.steps - 1,
                      self.hist_size + self.steps + self.max_future_steps) +
            start_location, target)
        predictions = (
            np.arange(self.hist_size + self.steps - 1,
                      self.hist_size + self.steps + self.max_future_steps) +
            start_location, preds)
        return historic, targets, predictions

    def plot_pred_future_multiple(
        self,
        num_plots: int,
    ):
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
            start_location = self.pred_start + i * self.max_future_steps
            if self.randomize_plot_location:
                start_location = np.random.randint(
                    self.amount_to_remove * 2,
                    len(self.data_valid) - self.amount_to_remove)
            historic, targets, predictions = self.get_future_plots(
                start_location)
            axi.set_title(f'Pred #{i}')
            axi.plot(historic[0], historic[1], label='hist')
            axi.plot(predictions[0], predictions[1], label='pred')
            axi.plot(targets[0], targets[1], label='target')
        fig.suptitle('Predict 2 hrs')
        fig.legend(['hist', 'pred', 'target'])
        plt.show()

    def plot_loss_history(self, model_name: str = "loss_plot"):
        """
        Plots the loss over time during training.
        """
        training_loss = self.history.history['loss']
        validation_loss = self.history.history['val_loss']
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
    imbalance_forecasting = ImbalanceForecasting("config/config1.ini")
    # imbalance_forecasting = ImbalanceForecasting("config/config2.ini")
    # imbalance_forecasting = ImbalanceForecasting("config/config3.ini")
    imbalance_forecasting.run()


if __name__ == "__main__":
    main()
