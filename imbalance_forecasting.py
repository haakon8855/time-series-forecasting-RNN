"""Haakon8855"""

import json
import numpy as np
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
        # Path to the weights of the model
        self.weights_path = f'models/{self.model_name}'
        # Number of epochs to train the network
        self.epochs = int(global_conf['epochs'])

        # Features to pass into the network
        self.cols_to_use = json.loads(global_conf['cols_to_use'])
        # Number of timesteps into the past the network can 'see'
        self.steps = int(global_conf['steps'])
        # Amount of rows to remove from the datasets
        self.amount_to_remove = int(global_conf['amount_to_remove'])
        # Maximum amount of steps to predict into the future
        self.max_future_steps = int(global_conf['max_future_steps'])
        # At which index to start predictions at
        self.pred_start = int(global_conf['pred_start'])
        # Number of timesteps to show before the prediction window
        self.hist_size = int(global_conf['hist_size'])

        # Whether to randomize the very last y_prev in each input to the network
        self.randomize_y_prev = global_conf['randomize_y_prev'] == 'True'
        # Whether to randomize the index to start predictions at
        self.randomize_plot_location = global_conf[
            'randomize_plot_location'] == 'True'
        # Whether to do the altered forecasting task or not
        self.altered_forecasting = global_conf['altered_forecasting'] == 'True'

        self.data_train = None
        self.data_valid = None
        self.data_holdout = None
        self.train_x = None
        self.train_y = None
        self.valid_x = None
        self.valid_y = None
        self.holdout_x = None
        self.holdout_y = None
        self.history = None

        # Initialize classes
        self.data_loader = DataLoader(self.altered_forecasting)
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
        self.data_holdout = self.data_loader.get_processed_data(
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
        self.holdout_x, self.holdout_y = DataLoader.strip_and_format_data(
            self.data_holdout,
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
        # Train the network (or load pretrained weights from file)
        # and store the history of the training (in order to plot
        # training loss and validation loss over time)
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

    def predict_one_step(self, limit: int = 300):
        """
        Predicts one step ahead and plots the resulting imbalance graph along
        with the target.
        """
        # Valid set
        self.plot_future_one_step(self.valid_x,
                                  self.valid_y,
                                  self.pred_start,
                                  limit,
                                  data_name='Validation')
        # Holdout set
        self.plot_future_one_step(self.holdout_x,
                                  self.holdout_y,
                                  self.pred_start,
                                  limit,
                                  data_name='Holdout')

    def predict_two_hours(self, num_plots_per_plot: int = 9):
        """
        Predicts two hours into the future by using the predicted values
        as y_prev for continued predictions.
        """
        self.plot_pred_future_multiple(self.data_valid,
                                       num_plots_per_plot,
                                       data_name='Validation')
        self.plot_pred_future_multiple(self.data_holdout,
                                       num_plots_per_plot,
                                       data_name='Holdout')

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

    def plot_future_one_step(self,
                             data_x,
                             data_y,
                             pred_start: int,
                             limit: int,
                             data_name: str = 'Unknown'):
        """
        Predicts one step into the future for 'limit' amount of timesteps.
        """
        test_x = data_x[pred_start:pred_start + limit]
        test_y = data_y[pred_start:pred_start + limit]
        y_pred = self.network.predict(test_x)
        x_axis = np.arange(pred_start, pred_start + limit)

        plt.figure(figsize=(15, 7))
        plt.title(f'{data_name} data set: Predict one step ahead')
        plt.plot(x_axis, test_y, label='target')
        plt.plot(x_axis, y_pred, label='pred')
        plt.legend()
        plt.show()

    def get_future_plots(self, data, start_location: int):
        """
        Predicts 'steps' amount of steps into the future. Using the previously
        predicted value as input for the next prediction.
        """
        # Where to start predictions
        preds_idx_start = self.amount_to_remove + start_location
        # Where to stop predictions
        preds_idx_stop = self.amount_to_remove + start_location + self.steps + self.max_future_steps
        # Get input to the network for predictions
        pred_input = data.iloc[preds_idx_start:preds_idx_stop][
            self.cols_to_use]
        # Get predictions
        preds = self.network.predict_into_future(pred_input)

        # Where the target plot starts and stops (usually 'steps' amount of timesteps
        # before where the predictions start, to show all information the
        # network gets).
        target_idx_start = self.amount_to_remove + start_location - self.hist_size
        target_idx_stop = preds_idx_stop
        # The target for the predictions in the whole time-frame
        target = data.iloc[target_idx_start:target_idx_stop]['y']

        if self.altered_forecasting:
            # Calculate the structured imbalance if running the altered
            # forecasting task. This calculated value will be later added
            # to the predictions.
            target_without = data.iloc[target_idx_start:target_idx_stop]['y']
            target = data.iloc[target_idx_start:target_idx_stop][
                'y_with_struct_imbal']
            target_struct_imbal = target - target_without

        # The historic target (target for the input data)
        hist = list(target[:self.hist_size + self.steps])
        last_hist = hist[-1]
        # Prediction window target (target for the predictions)
        target = target[self.hist_size + self.steps:self.hist_size +
                        self.steps + self.max_future_steps]
        if self.altered_forecasting:
            # Re-add the structural imbalance to the predictions to be able
            # to compare the values with the actual target value:
            target_struct_imbal = target_struct_imbal[self.hist_size + self.
                                                      steps:self.hist_size +
                                                      self.steps +
                                                      self.max_future_steps]
            preds += target_struct_imbal

        # Change datatype and prepend the last value from history
        # to create connected graphs.
        target = list(target)
        target.insert(0, last_hist)
        preds = list(preds)
        preds.insert(0, last_hist)

        # Create corresponding x-values for the x-axis and return values for plotting.
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

    def plot_pred_future_multiple(self,
                                  data,
                                  num_plots: int,
                                  data_name: str = 'Unknown'):
        """
        Predicts 'steps' amount of steps into the future. Using the previously
        predicted value as input for the next prediction.
        """
        rows = 3
        fig, axs = plt.subplots(nrows=rows,
                                ncols=(int(np.ceil(num_plots / rows))),
                                figsize=(15, 10))
        # Predicts 'num_plots' amount of future plots
        for i in range(len(axs.ravel())):
            axi = axs.ravel()[i]
            start_location = self.pred_start + i * self.max_future_steps
            # Get random start location
            if self.randomize_plot_location:
                start_location = np.random.randint(
                    self.amount_to_remove * 2,
                    len(data) - self.amount_to_remove * 2)
            # Get the x and y values of the graphs (historic, target and predictions)
            historic, targets, predictions = self.get_future_plots(
                data, start_location)
            # Pyplot stuff
            axi.set_title(f'Pred #{i}')
            axi.plot(historic[0], historic[1], label='hist')
            axi.plot(predictions[0], predictions[1], label='pred')
            axi.plot(targets[0], targets[1], label='target')
        fig.suptitle(f'{data_name} data set: Predict 2 hrs into future')
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
    # Best model
    forecast = ImbalanceForecasting("config/config1.ini")

    # Timeless model
    # forecast = ImbalanceForecasting("config/config2.ini")

    # 20 epoch model
    # forecast = ImbalanceForecasting("config/config3.ini")

    # struct_imbal as feature:
    # forecast = ImbalanceForecasting("config/config4.ini")

    # Altered forecasting target
    # forecast = ImbalanceForecasting("config/config5.ini")

    forecast.run()


if __name__ == "__main__":
    main()
