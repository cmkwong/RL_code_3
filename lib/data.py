import os
import csv
import glob
import numpy as np
from collections import OrderedDict
from lib import indicators

# Prices = collections.namedtuple('Prices', field_names=['open', 'high', 'low', 'close', 'volume'])

def float_available(f):
    if f == "null":
        return float(0)
    else:
        return float(f)

class csv_reader:
    def __init__(self):
        self.total_count_filter = 0
        self.total_count_out = 0
        self.total_count_fixed = 0

    def read_csv(self, file_name, sep='\t', filter_data=True, fix_open_price=False):
        data = {}
        print("Reading", file_name)
        with open(file_name, 'rt', encoding='utf-8') as fd:
            reader = csv.reader(fd, delimiter=sep)
            h = next(reader)
            if '<OPEN>' not in h and sep == ',':
                return self.read_csv(file_name, ';')
            indices = [h.index(s) for s in ('<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>')]
            date_list, o, h, l, c, v = [], [], [], [], [], []
            count_out = 0
            count_filter = 0
            count_fixed = 0
            prev_vals = None
            for row in reader:
                vals = [row[0]]
                row_data = list(map(float_available, [row[idx] for idx in indices]))
                vals.extend(row_data)
                if filter_data and ((vals[-1] < (1e-8))): # filter out the day when no volume
                    count_filter += 1
                    continue

                date, po, ph, pl, pc, pv = vals

                # fix open price for current bar to match close price for the previous bar
                if fix_open_price and prev_vals is not None:
                    ppo, pph, ppl, ppc, ppv = prev_vals
                    if abs(po - ppc) > 1e-8:
                        count_fixed += 1
                        po = ppc
                        pl = min(pl, po)
                        ph = max(ph, po)
                count_out += 1
                date_list.append(date)
                o.append(po)
                c.append(pc)
                h.append(ph)
                l.append(pl)
                v.append(pv)
                prev_vals = vals
        print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
            count_filter + count_out, count_filter, count_fixed))
        # stored data
        self.total_count_filter += count_filter
        self.total_count_out += count_out
        self.total_count_fixed += count_fixed
        # stacking
        data['open'] = np.array(o, dtype=np.float64)
        data['high'] = np.array(h, dtype=np.float64)
        data['low'] = np.array(l, dtype=np.float64)
        data['close'] = np.array(c, dtype=np.float64)
        data['volume'] = np.array(v, dtype=np.float64)
        return date_list, data

class SimpleSpliter:
    def __init__(self):
        self.trainSet_size = 0
        self.testSet_size = 0
        self.offset = 0

    def split_data(self, data, percentage=0.8):
        assert (isinstance(data, dict))
        train_data = {}
        test_data = {}
        self.offset = np.int(data['close'].shape[0] * percentage)
        for key in list(data.keys()):
            train_data[key] = data[key][:self.offset]
            test_data[key] = data[key][self.offset:]

        print("Split data done, training data: %d rows, eval data: %d" %
              (train_data['close'].shape[0], test_data['close'].shape[0]))
        self.trainSet_size += train_data['close'].shape[0]
        self.testSet_size += test_data['close'].shape[0]
        return train_data, test_data

def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result

def load_fileList(path):
    file_list = os.listdir(path)
    return file_list, path

def addition_indicators(prices, trend_names, status_names):
    trend_indicators = OrderedDict()
    status_indicators = OrderedDict()
    if trend_names is not None:
        for trend_name in trend_names:
            if trend_name == 'bollinger_bands':
                trend_indicators[trend_name] = indicators.Bollinger_Bands(prices, period=20, upperB_p=2, lowerB_p=2)
            if trend_name == 'MACD':
                trend_indicators[trend_name] = indicators.MACD(prices, period=(12,26), ma_p=9)
            if trend_name == 'RSI':
                trend_indicators[trend_name] = indicators.RSI(prices, period=14)
    if status_names is not None:
        for status_name in status_names:
            if status_name == 'RSI':
                status_indicators[status_name] = indicators.RSI(prices, period=14)
    return trend_indicators, status_indicators

def data_regularize(prices, trend_indicators, status_indicators):
    assert(isinstance(prices, dict))
    assert (isinstance(trend_indicators, dict))
    assert (isinstance(status_indicators, dict))
    train_set = {}
    test_set = {}
    # get required values from indicators
    for key in list(trend_indicators.keys()):
        trend_indicators[key].cal_data()
        required_data = trend_indicators[key].getData()
        prices.update(required_data)
    for key in list(status_indicators.keys()):
        status_indicators[key].cal_data()
        required_data = status_indicators[key].getData()
        prices.update(required_data)
    return prices

def update_cutoff(trend_indicators, status_indicators, offset):
    # update the cutoff offset for each indicators
    for key in list(trend_indicators.keys()):
        trend_indicators[key].cutoff = offset
    for key in list(status_indicators.keys()):
        status_indicators[key].cutoff = offset

def read_bundle_csv(path, sep=',', filter_data=True, fix_open_price=False, percentage=0.8, extra_indicator=False, trend_names=None, status_names=None):
    reader = csv_reader()
    spliter = SimpleSpliter()
    train_set = {}
    test_set = {}
    train_date = {}
    test_date = {}
    extra_set = {}
    file_list = os.listdir(path)
    for file_name in file_list:
        indicator_dicts = {} # extra_set = {"0005.HK": {"trend", "status"}, "0011.HK": {"trend", "status"}, ...}
        required_path = path + "/" + file_name
        dates, prices = reader.read_csv(required_path, sep=sep, filter_data=filter_data, fix_open_price=fix_open_price)
        if extra_indicator:
            indicator_dicts['trend'], indicator_dicts['status'] = addition_indicators(prices, trend_names, status_names)
            extra_set[file_name] = indicator_dicts
            # data regularize
            prices_ = data_regularize(prices, indicator_dicts['trend'], indicator_dicts['status'])
            train_set_, test_set_ = spliter.split_data(prices_, percentage=percentage)
            update_cutoff(indicator_dicts['trend'], indicator_dicts['status'], spliter.offset)
        else:
            train_set_, test_set_ = spliter.split_data(prices, percentage=percentage)
        train_set[file_name] = train_set_
        test_set[file_name] = test_set_
        train_date[file_name], test_date[file_name] = dates[:spliter.offset], dates[spliter.offset:]

    print("Totally, read done, got %d rows, %d filtered, %d open prices adjusted" % (
        reader.total_count_filter + reader.total_count_out, reader.total_count_filter, reader.total_count_fixed))
    print("The whole data set size for training: %d and for evaluation: %d" %(
        spliter.trainSet_size, spliter.testSet_size))

    return train_set, test_set, train_date, test_date, extra_set
