import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

import math
import random
import numpy as np
from scipy import stats
import time
from pdb import set_trace as b

import logger
from models import Model, Controller, NeuralNetEnv, TimeLimit
from learning_control.core.controllers import LQRController
from gym.envs.registration import register
import gym
from gym.wrappers import Monitor
from utls import ReplayBuffer, rollout
logger.info("[INFO] Finished Imports")


#################################
#################################
##                            ###
##       Model Fitting        ###
##                            ###
#################################
#################################

# Load data
data = ReplayBuffer(100000)
data.load('data.pkl')
logger.info("[INFO] Loaded Data")

# Initialize NN
num_ps = 1 # Learn 1 model
Ps = [Model() for _ in range(num_ps)] # can learn many models
logger.info("[INFO] Loaded Model(s)")

# Hyperparam
batch_size = 64
iterations = 50
epoch_per_iteration = 100

# Setup optimizer
optimizers = [optim.Adam(P.parameters(), lr=1e-3, eps=1e-8) for P in Ps]
logger.info("[INFO] Setup Optimizer(s)")

# Fit simulator
reset = True
for iteration in range(iterations):
    logger.info("[INFO] Starting interation %s" % iteration)

    for epoch in np.arange(epoch_per_iteration):
        losses, mses = [], []
        for P,optimizer in zip(Ps, optimizers):

            # Sample 2 batchess
            S,A,R,S_,D = data.sample(batch_size)

            with torch.no_grad():
                XA = torch.from_numpy(np.hstack([S, A])).float()
                X_prime = torch.from_numpy(S_).float()
            
            P_XA = P(XA)
            
            # MSE
            loss = (P_XA - X_prime).pow(2).sum(axis=1).mean()
           
            # Analytics
            losses.append(loss)
            mses.append([])

            # Learning step
            if loss == 0:
                logger.warn("[WARN] Loss is 0. Probably collapsed discriminator.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log analytics
        if epoch % 1 == 0:
            logger.info("Iteration {} | epoch = {}".format(iteration, epoch))
            for loss, mse in zip(losses,mses):
                logger.info("            |  loss = {}".format(float(loss.data.numpy())))
                # logger.info("            |  mse = {}".format(float(mse.data.numpy())))


#################################
#################################
##                            ###
##          Off-Policy        ###
##   Policy Evaluation Task   ###
##   using fitted model       ###
##                            ###
#################################
#################################

H = 200
seed = 1
thr = .5 # Not relevant for OPE
N = 100
gamma = .95

register(
    id='MBRLPend-v0',
    entry_point='envs.Pendulum:StochasticInvertedPendulum',
    max_episode_steps=H,
)


# Make environment
inner_env = gym.make('MBRLPend-v0', m=.25, l=.5)
env = Monitor(inner_env, './mle_videos', video_callable=False, force=True)

# Make simulator
simulator = TimeLimit(NeuralNetEnv(env, inner_env, Ps, thr=thr), H)

# Policy to evaluate
lqr = LQRController.build(env, Q=.1*np.identity(2), R=10*np.identity(1))
policy = Controller(env, lqr, seed=seed)

logger.info("[INFO] Deploying policy in environment. Collecting data")
# deploy policy in simulator
sim_data, _, _, sim_rewards = rollout(policy, simulator, N, epsilon=.3, no_vid=False, gamma=gamma)
# deploy policy live
actual_data, _, _, actual_rewards = rollout(policy, env, N, epsilon=.3, no_vid=False, gamma=gamma)

# Values
actual_value = np.mean(actual_rewards)
simulated_value = np.mean(sim_rewards)
behavior_value = data.value_of_data(gamma)

print()
logger.info("[INFO] Behavior V(pi): %s" % behavior_value)
logger.info("[INFO] Actual V(pi): %s" % actual_value)
logger.info("[INFO] Simulated V(pi): %s" % simulated_value)
logger.info("[INFO] Relative MSE: %s" % ((simulated_value - actual_value)**2 / (actual_value - behavior_value)**2))
import pdb; pdb.set_trace()