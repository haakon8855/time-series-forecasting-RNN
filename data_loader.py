"""haakon8855"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Loads the data from file and runs preprocessing on the dataset.
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_is_trained = False

    def get_processed_data(self, filepath):
        """
        Returns the dataframe loaded, preprocessed and with engineered features.
        """
        data = pd.read_csv(filepath)
        data = self.add_structural_imbalance(data)
        data = self.apply_preprocessing(data)
        data = self.apply_feature_engineering(data)
        return data

    def add_structural_imbalance(self, data):
        """
        Adds structural imbalance as a feature.
        """
        # Add load variable
        data['load'] = data['total'] - data['flow']

        # Get first index for where to start interpolation
        minutes = np.array(
            pd.to_datetime(data['start_time'].iloc[:12]).dt.minute)
        start_index = np.where(minutes == 30)[0][0]

        # x-values of load-column
        load_x = np.arange(0, len(data['load']))
        # x-values for each point in spline
        spline_load_x = np.arange(start_index, len(data['load']), 12)
        # y-values for each point in spline
        spline_load_y = np.array(data['load'].iloc[spline_load_x])

        # Fit spline on points
        tck = interpolate.splrep(spline_load_x, spline_load_y)
        # Get interpolated values of spline
        demand_y = interpolate.splev(load_x, tck)
        struct_imbal = data['load'] - demand_y
        data['struct_imbal'] = struct_imbal
        return data

    def apply_preprocessing(self, data):
        """
        Applies the necessary preprocessing steps to the loaded data.
        """
        # Create two copies of the original data. y_original will remain unchanged,
        # y and y* will be changed.
        data['y_original'] = data['y']
        # Winsorize one percent of the data
        data = self.winsorize_one_percent(data)
        # Normalize the data
        data = self.scale_data(data)
        # Add shifted y-value (y-val from yesterday and y-val from 5 mins ago)
        data = self.add_shifted_daily_value(data)
        data = self.shift_y(data)
        return data

    def apply_feature_engineering(self, data):
        """
        Apply the desired feature engineering steps.
        """
        data = self.add_hour_of_day(data)
        data = self.add_min_of_day(data)
        data = self.add_day_of_week(data)
        data = self.add_day_of_year(data)
        data = self.add_shifted_daily_mean(data)
        return data

    def winsorize_one_percent(self, data):
        """
        Winsorizes the data in the y-column. This is done by setting the upper
        0.5% and lower 0.5% to an upper and a lower bound.
        """
        winsorized = winsorize(data['y'], limits=[0.005, 0.005])
        data['y'] = np.array(winsorized)
        return data

    def scale_data(self, data):
        """
        Scales the given data using the rescaling method.
        """
        columns_to_scale = [
            'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
            'flow', 'y', 'struct_imbal'
        ]
        if not self.scaler_is_trained:
            self.scaler.fit(data[columns_to_scale])
            self.scaler_is_trained = True
        data[columns_to_scale] = self.scaler.transform(data[columns_to_scale])
        return data

    def shift_y(self, data):
        """
        Add shifted y value as y_prev.
        """
        data['y_prev'] = data['y'].shift(1)
        return data

    def add_hour_of_day(self, data):
        """
        Add hour of the day as a feature. 12 am is 0, 6 am is 6 and 1 pm is 13 ect.
        """
        hour_of_day = pd.to_datetime(data['start_time']).dt.hour
        data['hour_of_day'] = hour_of_day
        return data

    def add_min_of_day(self, data):
        """
        Add minute of day. Counting from 0 and up each minute after 00:00.
        Sine and cosine of minute of day is also added to have a smaller gap
        between 23:59 and 00:00.
        """
        min_of_day = pd.to_datetime(
            data['start_time']).dt.minute + data['hour_of_day'] * 60
        min_of_day_norm = 2 * np.pi * min_of_day / min_of_day.max()
        sin_minute = round(np.sin(min_of_day_norm), 6)
        cos_minute = np.cos(min_of_day_norm)

        data['min_of_day'] = min_of_day
        data['cos_minute'] = cos_minute
        data['sin_minute'] = sin_minute
        return data

    def add_day_of_week(self, data):
        """
        Add day of week. Counting from 0 (monday) up to 6 (sunday).
        Sine and cosine of day of week is also added to have smaller gap
        between sunday and monday.
        """
        day_of_week = pd.to_datetime(data['start_time']).dt.day_of_week
        day_of_week_norm = 2 * np.pi * day_of_week / day_of_week.max()
        sin_weekday = round(np.sin(day_of_week_norm), 2)
        cos_weekday = np.cos(day_of_week_norm)

        data['day_of_week'] = day_of_week
        data['cos_weekday'] = cos_weekday
        data['sin_weekday'] = sin_weekday
        return data

    def add_day_of_year(self, data):
        """
        Add day of year. Counting from 1 (1. jan) up to 365 (31. dec).
        Sine and cosine of day of year is also added to have smaller gap
        between 31. dec and 1. jan.
        """
        day_of_year = pd.to_datetime(data['start_time']).dt.day_of_year
        day_of_year_norm = 2 * np.pi * day_of_year / 365.25
        sin_yearday = np.sin(day_of_year_norm)
        cos_yearday = np.cos(day_of_year_norm)

        data['day_of_year'] = day_of_year
        data['cos_yearday'] = cos_yearday
        data['sin_yearday'] = sin_yearday
        return data

    def add_shifted_daily_value(self, data):
        """
        Add the value of y from 24hrs ago.
        """
        data['y_yesterday'] = data['y'].shift(288)
        return data

    def add_shifted_daily_mean(self, data):
        """
        Add shifted daily mean. Calculate the mean value of y from the day before
        and store it for each timestep in the current day.
        """
        data['date'] = pd.to_datetime(data['start_time']).dt.date
        daily_mean = data.groupby(['date']).y.mean()
        daily_mean = pd.DataFrame(daily_mean)
        daily_mean['date'] = daily_mean.index
        daily_mean = daily_mean.set_index(np.arange(len(daily_mean)))
        daily_mean = daily_mean.rename(columns={'y': 'daily_mean'})

        daily_mean['daily_mean'] = daily_mean['daily_mean'].shift(1)

        data = data.merge(daily_mean, left_on='date', right_on='date')
        return data

    @staticmethod
    def strip_and_split_x_y(data, cols_x, cols_y, amount_to_remove):
        """
        Removes the first n rows from the dataset, where n=amount_to_remove.
        This is to remove all rows containing nan values.
        """
        data_x_train_stripped = data.iloc[amount_to_remove:][cols_x]
        data_y_train_stripped = data.iloc[amount_to_remove:][cols_y]
        return data_x_train_stripped, data_y_train_stripped

    @staticmethod
    def format_input_data(data, steps, randomize_y_prev=False):
        """
        Returns the input data on the correct format.
        """
        formatted_data = []
        dataframe = data.copy()
        for i in range(len(dataframe.index) - steps + 1):
            formatted_data.append(dataframe.iloc[i:i + steps])
        formatted_data = np.array(formatted_data)
        if randomize_y_prev:
            for i in range(len(formatted_data)):
                formatted_data[i, -1, -7] = np.random.random() * 2 - 1
        return formatted_data

    @staticmethod
    def format_target_data(target, steps):
        """
        Returns the target data on the correct format.
        """
        return np.array(target[steps - 1:])

    @staticmethod
    def format_data(data_x, data_y, steps, randomize_y_prev=False):
        """
        Returns the data on the correct format.
        """
        train_x = DataLoader.format_input_data(data_x, steps, randomize_y_prev)
        train_y = DataLoader.format_target_data(data_y, steps)
        return train_x, train_y

    @staticmethod
    def strip_and_format_data(data,
                              cols_x,
                              cols_y,
                              amount_to_remove,
                              steps,
                              randomize_y_prev=False):
        """
        Strips the data from nan values and aranges it on the correct format
        for inputting to the network.
        """
        data_x_train_stripped, data_y_train_stripped = DataLoader.strip_and_split_x_y(
            data, cols_x, cols_y, amount_to_remove)
        train_x, train_y = DataLoader.format_data(data_x_train_stripped,
                                                  data_y_train_stripped, steps,
                                                  randomize_y_prev)
        return train_x, train_y


def main():
    """
    Main function of the data loader script.
    """
    loader_train = DataLoader()
    print(
        loader_train.get_processed_data('datasets/no1_train.csv').drop([
            'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
            'flow', 'min_of_day', 'day_of_week', 'cos_weekday', 'sin_weekday'
        ],
                                                                       axis=1))


if __name__ == "__main__":
    main()
