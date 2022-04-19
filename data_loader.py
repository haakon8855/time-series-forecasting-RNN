"""haakon8855"""

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Loads the data from file and runs preprocessing on the dataset.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.target_column = 'y_norm'
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        self.apply_preprocessing()
        self.apply_feature_engineering()

    def get_data(self):
        """
        Returns the dataframe loaded, preprocessed and with engineered features.
        """
        return self.data.copy()

    def apply_preprocessing(self):
        """
        Applies the necessary preprocessing steps to the loaded data.
        """
        # Create two copies of the original data. y_original will remain unchanged,
        # y and y* will be changed.
        self.data['y_original'] = self.data['y']
        # Winsorize one percent of the data
        self.winsorize_one_percent()
        # Normalize the data
        self.scale_data()
        # Add shifted y-value (y-val from yesterday and y-val from 5 mins ago)
        self.add_shifted_daily_value()
        self.shift_y()

    def apply_feature_engineering(self):
        """
        Apply the desired feature engineering steps.
        """
        self.add_hour_of_day()
        self.add_min_of_day()
        self.add_day_of_week()
        self.add_day_of_year()
        self.add_shifted_daily_mean()

    def winsorize_one_percent(self):
        """
        Winsorizes the data in the y-column. This is done by setting the upper
        0.5% and lower 0.5% to an upper and a lower bound.
        """
        winsorized = winsorize(self.data['y'], limits=[0.005, 0.005])
        self.data['y'] = np.array(winsorized)
        return self.data

    def scale_data(self):
        """
        Scales the given data using the rescaling method.
        """
        columns_to_scale = [
            'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
            'flow', 'y'
        ]
        self.scaler.fit(self.data[columns_to_scale])
        self.data[columns_to_scale] = self.scaler.transform(
            self.data[columns_to_scale])

    def shift_y(self):
        """
        Add shifted y value as y_prev.
        """
        self.data['y_prev'] = self.data['y'].shift(1)

    def add_hour_of_day(self):
        """
        Add hour of the day as a feature. 12 am is 0, 6 am is 6 and 1 pm is 13 ect.
        """
        hour_of_day = pd.to_datetime(self.data['start_time']).dt.hour
        self.data['hour_of_day'] = hour_of_day

    def add_min_of_day(self):
        """
        Add minute of day. Counting from 0 and up each minute after 00:00.
        Sine and cosine of minute of day is also added to have a smaller gap
        between 23:59 and 00:00.
        """
        min_of_day = pd.to_datetime(
            self.data['start_time']).dt.minute + self.data['hour_of_day'] * 60
        min_of_day_norm = 2 * np.pi * min_of_day / min_of_day.max()
        sin_minute = round(np.sin(min_of_day_norm), 6)
        cos_minute = np.cos(min_of_day_norm)

        self.data['min_of_day'] = min_of_day
        self.data['cos_minute'] = cos_minute
        self.data['sin_minute'] = sin_minute

    def add_day_of_week(self):
        """
        Add day of week. Counting from 0 (monday) up to 6 (sunday).
        Sine and cosine of day of week is also added to have smaller gap
        between sunday and monday.
        """
        day_of_week = pd.to_datetime(self.data['start_time']).dt.day_of_week
        day_of_week_norm = 2 * np.pi * day_of_week / day_of_week.max()
        sin_weekday = round(np.sin(day_of_week_norm), 2)
        cos_weekday = np.cos(day_of_week_norm)

        self.data['day_of_week'] = day_of_week
        self.data['cos_weekday'] = cos_weekday
        self.data['sin_weekday'] = sin_weekday

    def add_day_of_year(self):
        """
        Add day of year. Counting from 1 (1. jan) up to 365 (31. dec).
        Sine and cosine of day of year is also added to have smaller gap
        between 31. dec and 1. jan.
        """
        day_of_year = pd.to_datetime(self.data['start_time']).dt.day_of_year
        day_of_year_norm = 2 * np.pi * day_of_year / day_of_year.max()
        sin_yearday = np.sin(day_of_year_norm)
        cos_yearday = np.cos(day_of_year_norm)

        self.data['day_of_year'] = day_of_year
        self.data['cos_yearday'] = cos_yearday
        self.data['sin_yearday'] = sin_yearday

    def add_shifted_daily_value(self):
        """
        Add the value of y from 24hrs ago.
        """
        self.data['y_yesterday'] = self.data['y'].shift(288)

    def add_shifted_daily_mean(self):
        """
        Add shifted daily mean. Calculate the mean value of y from the day before
        and store it for each timestep in the current day.
        """
        self.data['date'] = pd.to_datetime(self.data['start_time']).dt.date
        daily_mean = self.data.groupby(['date']).y.mean()
        daily_mean = pd.DataFrame(daily_mean)
        daily_mean['date'] = daily_mean.index
        daily_mean = daily_mean.set_index(np.arange(len(daily_mean)))
        daily_mean = daily_mean.rename(columns={'y': 'daily_mean'})

        daily_mean['daily_mean'] = daily_mean['daily_mean'].shift(1)

        self.data = self.data.merge(daily_mean,
                                    left_on='date',
                                    right_on='date')


def main():
    """
    Main function of the data loader script.
    """
    loader_train = DataLoader('datasets/no1_train.csv')
    print(loader_train.get_data().drop([
        'hydro', 'micro', 'thermal', 'wind', 'river', 'total', 'sys_reg',
        'flow', 'min_of_day', 'day_of_week', 'cos_weekday', 'sin_weekday'
    ],
                                       axis=1))


if __name__ == "__main__":
    main()
