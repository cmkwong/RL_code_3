import os

from gym import wrappers
import ptan
import numpy as np
import torch
import torch.optim as optim

from lib import environ, data, models, common, validation

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# get the now time
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")

BATCH_SIZE = 32
BARS_COUNT = 20
TARGET_NET_SYNC = 1000
TRAINING_DATA = ""
VAL_DATA = ""

GAMMA = 0.99

REPLAY_SIZE = 100000 # 100000
REPLAY_INITIAL = 10000 # 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.00001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 0.9
EPSILON_STOP = 0.05
EPSILON_STEPS = 1000000
MAX_VALIDATION_EPISODES = 600

CHECKPOINT_EVERY_STEP = 50000
VALIDATION_EVERY_STEP = 30000 # 30000
WEIGHT_VISUALIZE_STEP = 50000

loss_v = None
load_net = True
TRAIN_ON_GPU = True

MAIN_PATH = "../docs/1"
DATA_LOAD_PATH = MAIN_PATH + "/data"
NET_SAVE_PATH = MAIN_PATH + "/checkpoint"
RECORD_SAVE_PATH = MAIN_PATH + "/records"
RUNS_SAVE_PATH = MAIN_PATH + "/runs/" + dt_string
NET_FILE = "checkpoint_13_3-1000000.data"

if __name__ == "__main__":
    if TRAIN_ON_GPU:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create the training, val set, trend_set, status_dicts
    train_set, val_set, train_date, val_date, extra_set = data.read_bundle_csv(
        path=DATA_LOAD_PATH,
        sep='\t', filter_data=True, fix_open_price=False, percentage=0.8, extra_indicator=True,
        trend_names=['bollinger_bands', 'MACD', 'RSI'], status_names=[])

    env = environ.StocksEnv(train_set, train_date, extra_set, bars_count=BARS_COUNT, reset_on_close=True, random_ofs_on_reset=True, volumes=False, train_mode=True)
    env = wrappers.TimeLimit(env, max_episode_steps=1000)
    env_val = environ.StocksEnv(val_set, val_date, extra_set, bars_count=BARS_COUNT, reset_on_close=True, random_ofs_on_reset=True, volumes=False, train_mode=False)
    # env_val = wrappers.TimeLimit(env_val, max_episode_steps=1000)

    # create neural network
    net = models.DoubleLSTM(price_input_size=env.price_size, trend_input_size=env.trend_size,
                            status_size=env.status_size, n_hidden=256, n_layers=2, rnn_drop_prob=0.2, fc_drop_prob=0.2,
                            actions_n=3, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)
    # load the network
    if load_net is True:
        with open(os.path.join(NET_SAVE_PATH, NET_FILE), "rb") as f:
            checkpoint = torch.load(f)
        net = models.DoubleLSTM(price_input_size=env.price_size, trend_input_size=env.trend_size,
                                status_size=env.status_size, n_hidden=256, n_layers=2, rnn_drop_prob=0.2, fc_drop_prob=0.2,
                                actions_n=3, train_on_gpu=TRAIN_ON_GPU, batch_first=True).to(device)
        net.load_state_dict(checkpoint['state_dict'])

    tgt_net = ptan.agent.TargetNet(net)

    # create buffer
    selector = ptan.actions.EpsilonGreedyActionSelector(EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, do_preprocess=False, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)

    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # create net pre-processor
    net_processor = common.netPreprocessor(net, tgt_net.target_model)

    # main training loop
    if load_net:
        step_idx = common.find_stepidx(NET_FILE, "-", "\.")
    else:
        step_idx = 0
    eval_states = None
    best_mean_val = None

    writer = SummaryWriter(log_dir=RUNS_SAVE_PATH, comment="USDJPY_2020")
    loss_tracker = common.lossTracker(writer, group_losses=100)
    with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            net_processor.populate_mode(batch_size=1)
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx*1.25 / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)
            if len(buffer) < REPLAY_INITIAL:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)

            # init the hidden both in network and tgt network
            net_processor.train_mode(batch_size=BATCH_SIZE)
            loss_v = common.calc_loss(batch, net, tgt_net.target_model, GAMMA ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            loss_value = loss_v.clone().detach().cpu().item()
            loss_tracker.loss(loss_value, step_idx)

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.sync()

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                # idx = step_idx // CHECKPOINT_EVERY_STEP
                checkpoint = {
                    "obs_space": env.observation_space.shape[0],
                    "action_n": env.action_space.n,
                    "state_dict": net.state_dict()
                }
                with open(os.path.join(NET_SAVE_PATH,"checkpoint-%d.data" % step_idx), "wb") as f:
                    torch.save(checkpoint, f)

            if step_idx % VALIDATION_EVERY_STEP == 0:
                net_processor.eval_mode(batch_size=1)
                validation_episodes = min(np.int((1/1800)*step_idx + 100), MAX_VALIDATION_EPISODES)
                writer.add_scalar("validation_episodes", validation_episodes, step_idx)

                val_epsilon = max(0, EPSILON_START - step_idx * 1.25 / EPSILON_STEPS)
                stats = validation.validation_run(env_val, net, episodes=validation_episodes,
                                                  save_path=RECORD_SAVE_PATH, step_idx=step_idx, epsilon=val_epsilon)
                common.valid_result_visualize(stats, writer, step_idx)

            if step_idx % WEIGHT_VISUALIZE_STEP == 0:
                net_processor.eval_mode(batch_size=1)
                common.weight_visualize(net, writer)