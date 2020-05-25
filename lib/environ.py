import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np
import collections

from . import data

DEFAULT_BARS_COUNT = 20
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Buy_close = 2


class State:
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True, train_mode=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes
        self.train_mode = train_mode
        self.order_step = 0.0 # for step counter from buy to sell (buy date = step 1, if sell date = 4, time cost = 3)
        self.bars_count_images = [80]  # [80 days, 200 days, ...]

    def reset(self, price, date, extra_set, offset):
        assert isinstance(price, dict)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._price = price
        self._date = date
        self._extra_set = extra_set     # empty if {}
        self.extra_indicator = False
        self._offset = offset

    def normalised_trend_data(self):
        start = self._offset - self.bars_count + 1
        end = self._offset + 1
        # normalise the data from an array
        x = 0
        y = 0
        target_data = np.ndarray(shape=(self.bars_count, self.extra_trend_size), dtype=np.float64)
        for indicator in self._extra_set['trend'].values():
            y = y + indicator.encoded_size
            target_data[:, x:y] = indicator.normalise(start, end, self.train_mode)
            x = y
            y = x
        return target_data

    def normalised_status_data(self): # disable
        pass

    def normalised_image_data(self, images):
        assert isinstance(images, list)
        for i, indicator in enumerate(self._extra_set['image'].values()):
            start = self._offset - self.bars_count_images[i] + 1
            end = self._offset + 1
            images[i] = indicator.normalise(start, end, self.train_mode)
        return images

    @property
    def shape_price(self):
        if self.volumes:
            return (self.bars_count, 5)
        else:
            return (self.bars_count, 4)

    @property
    def shape_trend(self):
        self.extra_trend_size = 0
        if len(self._extra_set) is not 0:
            if len(self._extra_set['trend']) is not 0:
                for trend_name in list(self._extra_set['trend'].keys()):
                    self.extra_trend_size += self._extra_set['trend'][trend_name].encoded_size
        return (self.bars_count, self.extra_trend_size)

    @property
    def shape_status(self):
        self.base_status_size = 2
        self.extra_status_size = 0
        if len(self._extra_set) is not 0:
            if len(self._extra_set['status']) is not 0:
                for status_name in list(self._extra_set['status'].keys()):
                    self.extra_status_size += self._extra_set['status'][status_name].encoded_size
        return (1, self.base_status_size + self.extra_status_size)

    @property
    def image_sizes(self):
        image_size_list = []
        if len(self._extra_set) is not 0:
            if len(self._extra_set['image']) is not 0:
                for i, image_name in enumerate(list(self._extra_set['image'].keys())):
                    image_feature_size = self._extra_set['image'][image_name].feature_size
                    image_size_list.append((self.bars_count_images[i], image_feature_size))
        return image_size_list

    def encode(self): # p.336
        """
        Convert current state into numpy array.
        """
        encoded_data = collections.namedtuple('encoded_data', field_names=['date', 'price', 'trend', 'image1', 'status'])
        price = np.ndarray(shape=self.shape_price, dtype=np.float32)
        trend = np.ndarray(shape=self.shape_trend, dtype=np.float32)
        status = np.ndarray(shape=self.shape_status, dtype=np.float32)
        # image shape
        image_shapes = self.image_sizes # list
        images = []
        for i, image_shape in enumerate(image_shapes):
            images.append(np.ndarray(shape=image_shape, dtype=np.float32))

        shift_r = 0
        # data stacking
        bese_volume = self._price['volume'][self._offset - self.bars_count + 1]
        for bar_idx in range(-self.bars_count + 1, 1):
            shift_c = 0
            price[shift_r, shift_c] = (self._price['high'][self._offset + bar_idx] - self._price['open'][self._offset + bar_idx]) / \
                                    self._price['open'][self._offset + bar_idx]
            shift_c += 1
            price[shift_r, shift_c] = (self._price['low'][self._offset + bar_idx] - self._price['open'][self._offset + bar_idx]) / \
                                    self._price['open'][self._offset + bar_idx]
            shift_c += 1
            price[shift_r, shift_c] = (self._price['close'][self._offset + bar_idx] - self._price['open'][self._offset + bar_idx]) / \
                                    self._price['open'][self._offset + bar_idx]
            shift_c += 1
            price[shift_r, shift_c] = (self._price['close'][(self._offset - 1) + bar_idx] - self._price['open'][self._offset + bar_idx]) / \
                                    self._price['open'][self._offset + bar_idx]
            shift_c += 1
            if self.volumes:
                price[shift_r, shift_c] = self._price['volume'][self._offset + bar_idx] / bese_volume
                shift_c += 1
            shift_r += 1
        # status stacking
        status[0,0] = (1/(1+399*np.exp((-self.order_step)/8))) - (1/400)
        if not self.have_position:
            status[0,1] = 0.0
        else:
            status[0,1] = (self._price['close'][self._offset] - self.open_price) / self.open_price

        # extra data
        if len(self._extra_set) is not 0:
            if len(self._extra_set['trend']) is not 0:
                trend = self.normalised_trend_data()
            if len(self._extra_set['status']) is not 0:
                pass
            if len(self._extra_set['image']) is not 0:
                images = self.normalised_image_data(images)

        # date
        date = self._date[self._offset]

        return encoded_data(date=date, price=price, trend=trend, image1=images[0], status=status)

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._price['open'][self._offset]
        rel_close = self._price['close'][self._offset]
        return open * (1.0 + rel_close)

    def time_cost(self,steps):
        one_step_cost = 0.2/(1+399*np.exp((-steps)/8)) # max 0.2% cost for a step holding
        return one_step_cost

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        # don't need self._cur_close() because it is not relative price
        close = self._price['close'][self._offset]
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        elif action == Actions.Buy_close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close                     # done if reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0
            self.order_step = 0.0

        self._offset += 1
        prev_close = close
        close = self._price['close'][self._offset]
        done |= self._offset >= self._price['close'].shape[0]-1 # done if reached limit

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close # change with respect to last day close-price
            self.order_step += 1
            one_step_cost = self.time_cost(self.order_step)  # cal the time cost
            reward -= one_step_cost

        return reward, done


class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price, date, extra_set, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False, train_mode=True):
        assert isinstance(price, dict)
        self.universe_price = price
        self.universe_date = date
        self.universe_extra_set = extra_set # empty dict if there is no extra data
        self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close, volumes=volumes, train_mode=train_mode)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.train_mode = train_mode
        self.seed()
        # get the shape first for creating the net
        self.get_data_shape()
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
        #                                        shape=(self._state.bars_count, self.data_size), dtype=np.float32)

    def get_data_shape(self):
        self.reset()
        self.price_size = self._state.shape_price
        self.trend_size = self._state.shape_trend
        self.image_sizes = self._state.image_sizes
        self.status_size = self._state.shape_status

    def offset_modify(self, prices, extra_set, train_mode):

        available_start = 0
        if len(extra_set) is not 0:
            # append the length, cal the min_length
            invalid_length = []
            if len(extra_set['trend']) is not 0:
                for key in list(extra_set['trend'].keys()):
                    invalid_length.append(extra_set['trend'][key].invalid_len)
            if len(extra_set['status']) is not 0:
                for key in list(extra_set['status'].keys()):
                    invalid_length.append(extra_set['status'][key].invalid_len)
            if len(extra_set['image']) is not 0:
                for key in list(extra_set['image'].keys()):
                    invalid_length.append(extra_set['image'][key].invalid_len)
            available_start = np.max(invalid_length)

        bars = max(self._state.bars_count, max(self._state.bars_count_images))
        if self.random_ofs_on_reset:
            if train_mode:
                offset = self.np_random.choice(range(available_start, prices['high'].shape[0] - bars * 10)) + bars
            else:
                offset = self.np_random.choice(prices['high'].shape[0] - bars * 10) + bars
        else:
            if train_mode:
                offset = bars + available_start
            else:
                offset = bars
        return offset

    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self.universe_price.keys()))
        price = self.universe_price[self._instrument]
        date = self.universe_date[self._instrument]
        extra_set_ = {}
        if len(self.universe_extra_set) is not 0:
            extra_set_ = self.universe_extra_set[self._instrument]
        offset = self.offset_modify(price, extra_set_, self.train_mode) # train_mode=False, random offset is different
        self._state.reset(price, date, extra_set_, offset)
        return self._state.encode()

    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]
