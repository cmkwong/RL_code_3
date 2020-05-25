import numpy as np
import pandas as pd

from lib import environ

class validator:
    def __init__(self, env, net, save_path, comission):
        self.env = env
        self.net = net
        self.save_path = save_path
        self.comission = comission

    def preparation(self, step_idx):
        self.stats = {
            'episode_reward': [],
            'episode_steps': [],
            'order_profits': [],
            'order_steps': []
        }
        self.df = pd.DataFrame(
            columns=['episode', 'buy_sell', 'order_date', 'order_position', 'close_date', 'close_position',
                     'position_steps', 'order_profit', "order_profit-timeCost"])
        self.df_row = pd.DataFrame(data=np.full((1, 9), ''),
                              columns=['episode', 'buy_sell', 'order_date', 'order_position', 'close_date',
                                       'close_position', 'position_steps', 'order_profit', "order_profit-timeCost"])
        self.path_csv = self.save_path + "/record_" + str(step_idx) + ".csv"
        # date_list for that env
        self.date_list = self.env._state._date

    def cal_profit(self, buy_sell=''):
        if buy_sell is 'buy_close':
            self.profit = (self.current_price - self.buy_position) - (
                        self.current_price + self.buy_position) * self.comission / 100
            self.profit = 100.0 * self.profit / self.buy_position
        elif buy_sell is 'sell_close':
            self.profit = (-1) * (self.current_price - self.sell_position) - (
                        self.current_price + self.sell_position) * self.comission / 100
            self.profit = 100.0 * self.profit / self.sell_position

    def update_dfrow_open(self, buy_sell=''):
        self.df_row['order_date'] = self.date_list[self.env._state._offset]
        if buy_sell is "buy":
            self.df_row['order_position'] = self.buy_position
        elif buy_sell is "sell":
            self.df_row['order_position'] = self.sell_position

    def update_dfrow_close(self, buy_sell='', episode=None):
        self.df_row['episode'] = episode
        if buy_sell is 'buy':
            self.df_row['buy_sell'] = 1
        elif buy_sell is 'sell':
            self.df_row['buy_sell'] = -1
        self.df_row['close_date'] = self.date_list[self.env._state._offset]
        self.df_row['close_position'] = self.current_price
        self.df_row['position_steps'] = self.position_steps
        self.df_row['order_profit'] = self.profit
        self.df_row['order_profit-timeCost'] = self.profit - self.time_cost

    def update_df(self):
        self.df = self.df.append(self.df_row)
        self.df_row[:] = ''  # clear the row

    def run(self, episodes, step_idx, epsilon):
        self.preparation(step_idx)

        for episode in range(episodes):
            obs = self.env.reset()

            self.total_reward = 0.0
            self.buy_position = None
            self.sell_position = None
            self.position_steps = None
            self.time_cost = 0.0
            self.episode_steps = 0

            while True:
                obs_v = [obs]
                out_v = self.net(obs_v)

                action_idx = out_v.max(dim=1)[1].item()
                if np.random.random() < epsilon:
                    action_idx = self.env.action_space.sample()
                action = environ.Actions(action_idx)

                self.current_price = self.env._state._price['close'][self.env._state._offset]  # base_offset = 8308

                if (action == environ.Actions.Buy) and (self.buy_position is None):
                    self.buy_position = self.current_price
                    self.position_steps = 0
                    # store the data
                    self.update_dfrow_open("buy")

                elif action == environ.Actions.Buy_close and self.buy_position is not None:
                    self.cal_profit('buy_close')
                    self.stats['order_profits'].append(self.profit)
                    self.stats['order_steps'].append(self.position_steps)

                    # store the data
                    self.update_dfrow_close('buy', episode=episode)
                    # stack into df
                    self.update_df()

                    # reset the value
                    self.buy_position = None
                    self.position_steps = None

                obs, reward, done, _ = self.env.step(action_idx)
                self.total_reward += reward
                self.episode_steps += 1
                if self.position_steps is not None:
                    self.position_steps += 1
                    self.time_cost += self.env._state.time_cost(self.position_steps)
                if done:
                    if self.buy_position is not None:
                        self.cal_profit('buy_close')
                        self.stats['order_profits'].append(self.profit)
                        self.stats['order_steps'].append(self.position_steps)

                        # store the data (have not sell yet but reached end-date)
                        self.update_dfrow_close('buy', episode=episode)
                        # stack into df and clear the df_row
                        self.update_df()
                    break
            self.stats['episode_reward'].append(self.total_reward)
            self.stats['episode_steps'].append(self.episode_steps)

            # export the csv files
        self.df.to_csv(self.path_csv, index=False)
        return self.stats
