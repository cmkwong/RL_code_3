import numpy as np
import pandas as pd

from lib import environ


def validation_run(env, net, episodes, save_path, step_idx, epsilon=0.02, comission=0.1):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': []
    }
    path_csv = save_path + "/record_" + str(step_idx) + ".csv"
    df = pd.DataFrame(
        columns=['episode', 'buy_sell', 'order_date', 'order_position', 'close_date', 'close_position','position_steps', 'order_profit', "profit_timeCost"])
    df_row = pd.DataFrame(data=np.full((1, 9), ''),
                          columns=['episode', 'buy_sell', 'order_date', 'order_position', 'close_date', 'close_position','position_steps', 'order_profit', "profit_timeCost"])
    # date_list for that env
    date_list = env._state._date

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        buy_position = None
        sell_position = None
        position_steps = None
        time_cost = 0.0
        episode_steps = 0

        while True:
            obs_v = [obs]
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            close_price = env._state._data['close'][env._state._offset] # base_offset = 8308

            if (action == environ.Actions.Buy) and (buy_position is None) and (sell_position is None):
                buy_position = close_price
                position_steps = 0
                # store the data
                df_row['order_date'] = date_list[env._state._offset]
                df_row['order_position'] = buy_position

            elif action == environ.Actions.Buy_close and buy_position is not None:
                profit = (close_price - buy_position) - (close_price + buy_position) * comission / 100
                profit = 100.0 * profit / buy_position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)

                # store the data
                df_row['episode'] = episode
                df_row['buy_sell'] = 1
                df_row['close_date'] = date_list[env._state._offset]
                df_row['close_position'] = close_price
                df_row['position_steps'] = position_steps
                df_row['order_profit'] = profit
                df_row['profit_timeCost'] = profit - time_cost
                # stack into df
                df = df.append(df_row)
                df_row[:] = ''    # clear the row

                # reset the value
                buy_position = None
                position_steps = None

            elif (action == environ.Actions.Sell) and (buy_position is None) and (sell_position is None):
                sell_position = close_price
                position_steps = 0
                # store the data
                df_row['order_date'] = date_list[env._state._offset]
                df_row['order_position'] = sell_position

            elif action == environ.Actions.Sell_close and sell_position is not None:
                profit = (-1) * (close_price - sell_position) - (close_price + sell_position) * comission / 100
                profit = 100.0 * profit / sell_position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)

                # store the data
                df_row['episode'] = episode
                df_row['buy_sell'] = -1
                df_row['close_date'] = date_list[env._state._offset]
                df_row['close_position'] = close_price
                df_row['position_steps'] = position_steps
                df_row['order_profit'] = profit
                df_row['profit_timeCost'] = profit - time_cost
                # stack into df
                df = df.append(df_row)
                df_row[:] = ''  # clear the row

                # reset the value
                sell_position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
                time_cost += env._state.time_cost(position_steps)
            if done:
                if buy_position is not None:
                    profit = (close_price - buy_position) - (close_price + buy_position) * comission / 100
                    profit = 100.0 * profit / buy_position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)

                    # store the data (have not sell yet but reached end-date)
                    df_row['episode'] = episode
                    df_row['buy_sell'] = 1
                    df_row['close_date'] = date_list[env._state._offset]
                    df_row['close_position'] = close_price
                    df_row['position_steps'] = position_steps
                    df_row['order_profit'] = profit
                    df_row['profit_timeCost'] = profit - time_cost
                    # stack into df and clear the df_row
                    df = df.append(df_row)
                    df_row[:] = ''

                elif sell_position is not None:
                    profit = (-1) * (close_price - sell_position) - (close_price + sell_position) * comission / 100
                    profit = 100.0 * profit / sell_position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)

                    # store the data (have not sell yet but reached end-date)
                    df_row['episode'] = episode
                    df_row['buy_sell'] = -1
                    df_row['close_date'] = date_list[env._state._offset]
                    df_row['close_position'] = close_price
                    df_row['position_steps'] = position_steps
                    df_row['order_profit'] = profit
                    df_row['profit_timeCost'] = profit - time_cost
                    # stack into df and clear the df_row
                    df = df.append(df_row)
                    df_row[:] = ''
                break
        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    # export the csv files
    df.to_csv(path_csv, index=False)
    return stats

