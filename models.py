import torch
import torch.nn as nn
from learning_control.core.controllers import FBLinController
import numpy as np
import gym

# Simulator, Model definition.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

    def forward(self, XA):
        output = self.net(XA)
        return output

class Controller(FBLinController):
    """docstring for Controller"""
    def __init__(self, *args, seed=10, **kw):
        super(Controller, self).__init__(*args, **kw)
        self.random = np.random.RandomState(seed)
        self.uerr = 1e-3

    def noise(self, mean, std):
        return self.random.normal(mean, std**2)

    def act(self, x, epsilon):
        t = 0
        u = self.eval(x, t) + self.noise(0, self.uerr)
        u = self.process(u)
        if self.random.uniform() > epsilon:
            return u
        else:
            return np.atleast_1d(self.random.uniform(-2,2))

class NeuralNetEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, env, inner_env, dynamics, thr=.2):
        super().__init__()
        self.vectorized = False
        self.env = env
        self._spec = inner_env.spec # copy spec
        self.inner_env = inner_env
        self.is_done = getattr(inner_env, 'is_done', lambda x, y: np.asarray([False] * len(x)))
        self.dynamics = dynamics if isinstance(dynamics, list) else [dynamics]
        self.thr = thr

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self.env.spec

    @spec.setter
    def set_spec(self, spec):
        self.spec = spec

    def render(self, *args, **kwargs):
        return self.env.render(mode='rgb_array', state=self.state)
        # arr = (im * 255).astype('uint8')[0][0]
        # return np.concatenate([np.expand_dims(arr,-1)] * 3, -1)
        # return self.env.render(*args, **kwargs)

    def reset(self):
        self.state = self.env.reset()
        observation = np.copy(self.state)
        return observation

    def step(self, action, use_states=None):
        # action = np.clip(action, *self.action_space.bounds)
        if use_states is not None:
            # next_observation = self.dynamics.forward([use_states], [action])[0]
            # obs_dim = self.env.observation_space.shape[0]
            # next_observation[:obs_dim] = np.clip(next_observation[:obs_dim], *self.observation_space.bounds)
            # next_observation[:obs_dim] = np.clip(next_observation[:obs_dim], -1e5, 1e5)
            with torch.no_grad():
                XA = torch.from_numpy(np.hstack([use_states, action])).float()
                next_observations = [P.forward(XA).numpy() for P in self.dynamics]
        else:
            with torch.no_grad():
                # np.expand_dims(self.state,0), np.array([action])
                X = torch.from_numpy(np.expand_dims(self.state,0)).float()
                A = torch.from_numpy(np.atleast_1d(action)).long()
                # XA = torch.from_numpy(np.hstack([np.expand_dims(self.state,0), np.eye(3)[np.atleast_1d(action)]])).float()
                XA = torch.from_numpy(np.hstack([self.state, action])).float()
                next_observations = [P.forward(XA).numpy() for P in self.dynamics]
            # next_observation = np.clip(next_observation, *np.array([[-1.2,-.07],[.6,.07]]))
            # next_observation = np.clip(next_observation, -1e5, 1e5)

        # MOREL
        if len(next_observations) > 1:
            next_observations = np.array(next_observations)
            next_observation = next_observations.mean(axis=0)
            disc = np.linalg.norm(next_observations - next_observation, axis=1).max()
        else:
            next_observation = next_observations[0]

        if hasattr(self.inner_env, "env"):
            reward = - self.inner_env.env.cost_np_vec(np.array(self.state)[None], np.array(action)[None], np.array([next_observation]))[0]
        else:
            reward = - self.inner_env.cost_np_vec(np.array(self.state)[None], np.array(action)[None], np.array([next_observation]))[0]

        done = self.is_done(np.array(self.state)[None], np.array(action)[None], next_observation)[0]
        self.state = np.reshape(next_observation, -1)

        if (len(self.dynamics)>1) and (disc > self.thr):
            done = True
            reward = np.array([-100])

        return self.state, reward, done, {'goal_reached':done} #self.inner_env.step({"next_obs": next_observation, "reward": reward, "done": done})

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
