import numpy as np
import pandas as pd

from lib import environ


def validation_run(env, net, episodes, save_path, step_idx, epsilon=0.02, comission=0.1, print_out=False):
    stats = {
        'episode_reward': [],
        'episode_steps': [],
        'order_profits': [],
        'order_steps': []
    }
    path_csv = save_path + "/price_" + str(step_idx) + ".csv"
    df = pd.DataFrame(
        columns=['episode', 'buy_date', 'buy_position', 'sell_date', 'sell_position','position_steps', 'order_profit'])
    df_row = pd.DataFrame(data=np.full((1, 7), ''),
                          columns=['episode', 'buy_date', 'buy_position', 'sell_date', 'sell_position','position_steps', 'order_profit'])
    # date_list for that env
    date_list = env._state._date

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        position = None
        position_steps = None
        episode_steps = 0

        while True:
            obs_v = [obs]
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = environ.Actions(action_idx)

            close_price = env._state._data['close'][env._state._offset] # base_offset = 8308

            if action == environ.Actions.Buy and position is None:
                position = close_price
                position_steps = 0
                # store the data
                df_row['buy_date'] = date_list[env._state._offset]
                df_row['buy_position'] = position
                # print it out
                if print_out:
                    print("Buy -Date: %s  -position: %.3f  -total_reward(c): %.3f"
                          % (date_list[env._state._offset], position, total_reward))
            elif action == environ.Actions.Close and position is not None:
                profit = close_price - position - (close_price + position) * comission / 100
                profit = 100.0 * profit / position
                stats['order_profits'].append(profit)
                stats['order_steps'].append(position_steps)

                # store the data
                df_row['episode'] = episode
                df_row['sell_date'] = date_list[env._state._offset]
                df_row['sell_position'] = close_price
                df_row['position_steps'] = position_steps
                df_row['order_profit'] = profit
                # stack into df and clear the df_row
                df = df.append(df_row)
                df_row[:] = ''
                # print it out
                if print_out:
                    print("Sell -Date: %s  -position: %.3f  -total_reward(c): %.3f  -steps: %d -order profit: %.3f"
                          % (date_list[env._state._offset], close_price, total_reward, position_steps, profit))
                # reset the value
                position = None
                position_steps = None

            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    profit = close_price - position - (close_price + position) * comission / 100
                    profit = 100.0 * profit / position
                    stats['order_profits'].append(profit)
                    stats['order_steps'].append(position_steps)

                    # store the data (have not sell yet but reached end-date)
                    df_row['episode'] = episode
                    df_row['sell_date'] = date_list[env._state._offset]
                    df_row['sell_position'] = close_price
                    df_row['position_steps'] = position_steps
                    df_row['order_profit'] = profit
                    # stack into df and clear the df_row
                    df = df.append(df_row)
                    df_row[:] = ''
                    # print it out
                    if print_out:
                        print("End -Date: %s  -position: %.3f  -total_reward(c): %.3f  -steps: %d -order profit: %.3f"
                              % (date_list[env._state._offset], close_price, total_reward, position_steps, profit))
                break
        stats['episode_reward'].append(total_reward)
        stats['episode_steps'].append(episode_steps)

    # export the csv files
    df.to_csv(path_csv, index=False)
    return stats

