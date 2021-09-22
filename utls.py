import numpy as np
from collections import deque
import random
import pickle


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.initial_states = []

    def push(self, state, action, reward, next_state, done, initial=False):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        if initial:
            self.initial_states.append(state)

        self.buffer.append((state, np.array(action), reward, next_state, done))
    
    def value_of_data(self, gamma):
        t = 0
        total = 0
        n_traj = 0
        for (_,_,cost,_,done) in self.buffer:
            total += cost[0] * gamma**t

            t += 1
            if done: 
                t = 0
                n_traj += 1
        # must have at least 1 full trajectory.
        return total/n_traj

    def sample(self, batch_size):
        N = min(batch_size, self.__len__())
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, N))
        return np.concatenate(state), np.array(action), np.array(reward), np.concatenate(next_state), np.array(done)

    def get_initial_states(self):
        return np.concatenate(self.initial_states)

    def get(self, N, idx):
        return np.array([self.buffer[i][idx] for i in range(N)])

    def __len__(self):
        return len(self.buffer)
    
    def save(self, name):
        with open(name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    def load(self, name):
        with open(name, 'rb') as input:
            tmp = pickle.load(input)
        self.buffer = tmp.buffer
        self.initial_states = tmp.initial_states

def rollout(current_model, env, N, epsilon=.05, no_vid=False, gamma=.99):
    data = ReplayBuffer(100000)
    goals = []
    episode_rewards = []
    lengths = []
    for idx in range(N):
        done = False
        goal = 0
        episode_reward = 0
        episode_length = 0
        # import pdb; pdb.set_trace()
        state = env.reset()
        initial = True
        while not done:
            action = current_model.act(state, epsilon)

            next_state, reward, done, dic = env.step(action)
            data.push(state, action, reward, next_state, done, initial=initial)
            initial = False
            state = next_state
            episode_reward += reward * gamma**episode_length
            goal += dic['goal_reached']
            episode_length += 1
        goals.append(goal)
        lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        print(episode_reward, episode_length, goal)
    print(np.mean(goals), np.mean(lengths), np.mean(episode_rewards))
    return data, goals, lengths, episode_rewards



