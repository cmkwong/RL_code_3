import copy
import numpy as np
import torch
# import torch.nn.functional as F

# from . import actions


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError

def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector):
        self.dqn_model = dqn_model
        self.action_selector = action_selector

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states

class Supervised_DQNAgent(BaseAgent):
    def __init__(self, dqn_model, action_selector, sample_sheet, assistance_ratio=0.2):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.sample_sheet = sample_sheet # name tuple
        self.assistance_ratio = assistance_ratio

    def __call__(self, states, agent_states=None):
        batch_size = len(states)
        if agent_states is None:
            agent_states = [None] * batch_size
        sample_mask = np.random.random(batch_size) <= self.assistance_ratio
        sample_actions_ = []
        dates = [state.date for state in states[sample_mask]]
        for date in dates:
            for i, d in enumerate(self.sample_sheet.date):
                if d == date:
                    sample_actions_.append(self.sample_sheet.action[i])
        sample_actions = np.array(sample_actions_)   # convert into array

        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        actions[sample_mask] = sample_actions
        return actions, agent_states

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
