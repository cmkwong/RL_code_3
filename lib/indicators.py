import pandas as pd
import numpy as np

class Bollinger_Bands:
    def __init__(self, target_price, period=20, upperB_p=2, lowerB_p=2):
        self.target_price = target_price
        self.period = period
        self.upperB_p = upperB_p
        self.lowerB_p = lowerB_p
        self.size = 2
        self.encoded_size = 1
        self.invalid_len = self.period - 1
        self.cutoff = None

    def cal_data(self):
        # close price
        close_series = pd.Series(self.target_price['close'])
        # cal the SMA
        self.SMA = np.array(close_series.rolling(self.period).mean(), dtype=np.float64)
        # std
        std = np.array(close_series.rolling(self.period).std(ddof=0), dtype=np.float64)
        # cal the upper bond
        self.upperBand = self.SMA + (std * self.upperB_p)
        # cal the lower bond
        self.lowerBand = self.SMA - (std * self.lowerB_p)

    def getData(self):
        return {'upperBand': self.upperBand, 'lowerBand': self.lowerBand}

    def normalise(self, start, end, train_mode):
        if train_mode is False:
            start = start + self.cutoff
            end = end + self.cutoff
        # assert(isinstance(data_array, np.ndarray))
        target_data = (self.target_price['close'][start:end] - self.lowerBand[start:end]) / \
                      (self.upperBand[start:end] - self.lowerBand[start:end])
        return target_data.reshape(-1, self.encoded_size)

class MACD:
    def __init__(self, target_price, period=(12, 26), ma_p=9):
        self.target_price = target_price
        self.period = period
        self.ma_p = ma_p
        self.size = 2
        self.encoded_size = 2
        self.invalid_len = self.period[1] + self.ma_p - 2
        self.cutoff = None

    def cal_data(self):
        assert (self.period[1] > self.period[0])
        macd_value_1 = 2 / (self.period[0] + 1)
        macd_value_2 = 2 / (self.period[1] + 1)
        macd_value_3 = 2 / (self.ma_p + 1)

        # create empty array
        EMA_1_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        EMA_2_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        MACD_fast_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        MACD_fast_array_n = MACD_fast_array.copy()
        MACD_slow_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        MACD_slow_array_n = MACD_slow_array.copy()

        # cal EMA_1_array
        for idx in range((self.period[0] - 1), EMA_1_array.shape[0]):
            if idx is (self.period[0] - 1):
                EMA_1_array[idx] = self.target_price['close'][:self.period[0]].mean()
            else:
                EMA_1_array[idx] = (self.target_price['close'][idx] - EMA_1_array[idx - 1]) * macd_value_1 + EMA_1_array[
                    idx - 1]

        # cal EMA_2_array & cal MACD_fast
        for idx in range((self.period[1] - 1), EMA_2_array.shape[0]):
            if idx is (self.period[1] - 1):
                EMA_2_array[idx] = self.target_price['close'][:self.period[1]].mean()
            else:
                EMA_2_array[idx] = (self.target_price['close'][idx] - EMA_2_array[idx - 1]) * macd_value_2 + EMA_2_array[
                    idx - 1]
            MACD_fast_array[idx] = EMA_1_array[idx] - EMA_2_array[idx]
            MACD_fast_array_n[idx] = ((EMA_1_array[idx] - EMA_2_array[idx]) / EMA_2_array[idx]) * 100

        # cal MACD_slow
        for idx in range((self.period[1] + self.ma_p - 2), MACD_slow_array.shape[0]):
            if idx is (self.period[1] + self.ma_p - 2):
                MACD_slow_array[idx] = MACD_fast_array[idx - self.ma_p + 1:idx + 1].mean()
                MACD_slow_array_n[idx] = MACD_fast_array_n[idx - self.ma_p + 1:idx + 1].mean()
            else:
                MACD_slow_array[idx] = (MACD_fast_array[idx] - MACD_slow_array[idx - 1]) * macd_value_3 + \
                                       MACD_slow_array[idx - 1]
                MACD_slow_array_n[idx] = (MACD_fast_array_n[idx] - MACD_slow_array_n[idx - 1]) * macd_value_3 + \
                                       MACD_slow_array_n[idx - 1]
        # store the standard data
        self.EMA_1 = EMA_1_array
        self.EMA_2 = EMA_2_array
        self.MACD_fast = MACD_fast_array
        self.MACD_slow = MACD_slow_array
        # store the normalised data
        self.MACD_fast_n = MACD_fast_array_n
        self.MACD_slow_n = MACD_slow_array_n

    def getData(self):
        return {'MACD_fast': self.MACD_fast, 'MACD_slow': self.MACD_slow}

    def normalise(self, start, end, train_mode):
        if train_mode is False:
            start = start + self.cutoff
            end = end + self.cutoff
        # assert(isinstance(data_array, np.ndarray))
        target_data = np.ndarray(shape=((end - start), self.encoded_size), dtype=np.float64)
        target_data[:, 0] = self.MACD_fast_n[start:end]
        target_data[:, 1] = self.MACD_slow_n[start:end]
        return target_data.reshape(-1, self.encoded_size)

class RSI:
    def __init__(self, target_price, period):
        self.target_price = target_price
        self.period = period
        self.size = 1
        self.encoded_size = 1
        self.invalid_len = self.period
        self.cutoff = None

    def cal_data(self):
        rsi = {}
        # cal the change
        change = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        change[1:] = np.diff(self.target_price['close'])
        self.change = change

        # cal the upward_movement
        mask_positive = self.change > 0
        upward_movement = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        upward_movement[mask_positive] = self.change[mask_positive]
        self.upward_movement = upward_movement

        # cal the upward_movement
        mask_negative = self.change < 0
        downward_movement = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        downward_movement[mask_negative] = -self.change[mask_negative]
        self.downward_movement = downward_movement

        # cal averg_upward_movement & averg_downward_movement
        averg_upward_movement_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        averg_downward_movement_array = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        averg_upward_movement_array[self.period] = self.upward_movement[1:self.period + 1].mean()
        averg_downward_movement_array[self.period] = self.downward_movement[1:self.period + 1].mean()

        for idx in range((self.period + 1), averg_upward_movement_array.shape[0]):
            averg_upward_movement_array[idx] = (averg_upward_movement_array[idx - 1] * (self.period - 1) +
                                                self.upward_movement[idx]) / self.period
            averg_downward_movement_array[idx] = (averg_downward_movement_array[idx - 1] * (self.period - 1) +
                                                  self.downward_movement[idx]) / self.period
        self.averg_upward_movement = averg_upward_movement_array
        self.averg_downward_movement = averg_downward_movement_array

        # cal relative strength
        relat_strength = np.zeros(shape=(self.target_price['close'].shape[0],), dtype=np.float64)
        relat_strength[self.period:] = self.averg_upward_movement[self.period:] / self.averg_downward_movement[
                                                                                  self.period:]
        self.relative_strength = relat_strength

        # cal RSI value
        self.rsi_value = 100 - 100 / (self.relative_strength + 1)
    def getData(self):
        return {'rsi': self.rsi_value}

    def normalise(self, start, end, train_mode):
        if train_mode is False:
            start = start + self.cutoff
            end = end + self.cutoff
        # assert(isinstance(data_array, np.ndarray))
        # target_data = np.ndarray(shape=((end-start), self.encoded_size),dtype=np.float64)
        target_data = self.rsi_value[start:end] / 100
        return target_data.reshape(-1,self.encoded_size)


class moving_average:  # 2d_data
    def __init__(self, target_price, periods):
        self.target_price = target_price
        self.periods = periods  # [1, 2,3,4,5,6,7,8,...,100]
        self.feature_size = len(periods)
        self.invalid_len = max(periods)-1
        self.cutoff = None

    def cal_data(self):
        self.mas = np.zeros(shape=(self.target_price['close'].shape[0], len(self.periods)), dtype=np.float32)
        for c, period in enumerate(self.periods):
            self.mas[:, c] = pd.Series(self.target_price['close']).rolling(period).mean().values

    def getData(self):
        return {'ma': self.mas}

    def normalise(self, start, end, train_mode):
        if train_mode is False:
            start = start + self.cutoff
            end = end + self.cutoff
        min = np.min(self.mas[start:end, :])
        max = np.max(self.mas[start:end, :])
        target_data = (self.mas[start:end, :] - min) / (max - min)
        return target_data  # 2d image (20,100)