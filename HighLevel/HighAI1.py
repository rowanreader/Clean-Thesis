import gym
from Utils import plot_learning_curve, plot_average_curve
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import torch.nn as nn
import pickle
from HighEnvs import HighEnv1
import numpy as np
import HighTowerSim as tower
from HighTowerSim import Tile
import os
import warnings
from stable_baselines3.common.callbacks import EvalCallback
from collections import defaultdict
import plottingDennyBritz as plotting
import itertools
import matplotlib
import matplotlib.style
matplotlib.style.use('ggplot')
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
# from stable_baselines.common import make_vec_env
# from stable_baselines import ACER

# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore") # get rid of warnings


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def eGreedy(Q, e, num):
    # e-greedy based on Q table

    def policy(state):
        actionProb = np.ones(num, dtype=float)*e/num
        best = np.argmax(Q[state]) # get max q value index
        actionProb[best] += 1 - e   # NOT SURE !!!!!
        return actionProb

    return policy

def QLearning(env, episodes, gamma = 0.99, alpha = 0.6, e = 0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # track stats
    stats = plotting.EpisodeStats(episode_lengths = np.zeros(episodes), episode_rewards = np.zeros(episodes))
    policy = eGreedy(Q, e, env.action_space.n)

    for i in range(episodes):
        state = env.reset()
        for t in itertools.count():

            probabilities = policy(state)
            # select action based on policy probabilities
            action = np.random.choice(np.arange(len(probabilities)), p = probabilities)

            # take action and get reward
            newState, reward, done, _ = env.step(action)

            stats.episode_lengths[i] = t
            stats.episode_rewards[i] += reward

            # TD update
            nextBestAction = np.argmax(Q[newState])
            tdTarget = reward + gamma*Q[newState][nextBestAction]
            tdDelta = tdTarget - Q[state][action]
            Q[state][action] += alpha*tdDelta

            if done:
                break
            state = newState
    return Q, stats



if __name__ == "__main__":
    # run = "HighAI1"
    # premodelName = "DQN_100Towers_"
    premodelName = "All_OrderedSetPillars"
    modelName = "All_OrderedRandPillars"
    # modelName = "Rand"
    # modelName = "DQN_RandPillars"
    # env = HighEnv1()
    # env = Monitor(env)
    # # # Q Learning
    # # Q, stats = QLearning(env, 5000)
    # # fig1, fig2, fig3 = plotting.plot_episode_stats(stats)
    # #
    # # fig1.savefig("Fig1.png")
    # #
    # # fig2.savefig("Fig2.png")
    # #
    # # fig3.savefig("Fig3.png")
    # # env = gym.make("CartPole-v0")
    #
    #
    # n_actions = env.action_space.n
    # print(n_actions)
    # policy_kwargs = dict(net_arch=[128, 128], activation_fn=th.nn.modules.activation.LeakyReLU)
    # model = DQN.load(premodelName, env=env, learning_rate=3e-5)
    # #
    # # # model = DQN("MlpPolicy", env, verbose=1, learning_starts=150, exploration_fraction=0.9, policy_kwargs=policy_kwargs)
    # # # model = DQN("CustomNetwork", env, learning_rate=0.015, verbose=1, learning_starts=50, policy_kwargs=policy_kwargs, exploration_fraction=0.65)
    # eval_callback = EvalCallback(env,  best_model_save_path=modelName + "_best",
    #                              log_path='./logs/', eval_freq=150,
    #                              render=False)
    # model.learn(total_timesteps=int(3e6), log_interval=150, callback=eval_callback)
    #
    # model.save(modelName)
    # print("Saved!\n")
    #
    # temp = env.get_episode_rewards()
    # n_games = len(temp)
    # x = [i+1 for i in range(n_games)]
    # figure_file = modelName + "Rewards.png"
    # title = modelName
    # yAx = "Rewards"
    # plot_average_curve(x, temp, title, figure_file, yAx)
    #
    # yAx = "Number of Illegal Moves"
    # illegals = env.getIllegal()
    # n_games = len(illegals)
    # x = [i + 1 for i in range(n_games)]
    # figure_file = modelName + "Illegal.png"
    # title = modelName + "Illegal"
    # plot_average_curve(x, illegals, title, figure_file, yAx)
    # # file = open("EpisodeRewards.txt", "w")
    # # file.write(str(temp))
    # # file.close()
    # del model  # remove to demonstrate saving and loading


    # now apply to testing

    # fileName = "TestTileTower.txt"
    fileName = "TestTowerSim_100.txt"
    num = 300  # 126 # how many towers are in the file
    # env = HighEnv1(fileName)
    env = HighEnv1()

    model = DQN.load(modelName + "_best/best_model", env=env)  # + "_best/best_model")

    totalRewards = []
    illegalMoves = []
    winArray = []
    score = 0
    count = 1
    scores = []
    yAx = "Rewards"
    # print("Episode 0")
    # print(model.policy.net_arch)
    # print(model.policy.action_space.n)
    for i in range(100):
        env = HighEnv1()
        env = Monitor(env)
        # obs = env.reset()
        scores = []
        count = 0
        for j in range(num):
            dones = False
            obs = env.reset()
            count += 1

            print("Episode:", count)
            while not dones:
                action, _states = model.predict(np.reshape(obs, (38,)))
                obs, rewards, dones, info = env.step(action)
                score += rewards

            scores.append(score)
            score = 0
            if count > num:
                print("Oops")
                break


        # finished 25 towers
        totalRewards.append(scores)
        k = env.getIllegal()
        illegalMoves.append(k)
        wins = [1 if x <= 5 else 0 for x in k]
        winArray.append(wins)

    x = [i for i in range(len(scores))]
    toPlot = np.mean(totalRewards, axis = 0)
    title = "Validation Pillar Placement AI for " + str(num) + " Towers " + "best policy"
    figure_file = modelName + "Validate Reward Average.png"
    plot_learning_curve(x, scores, title, figure_file, yAx)

    n_games = len(wins)
    plotRewards = np.mean(winArray, axis=0)
    print(np.sum(plotRewards))
    x = [i + 1 for i in range(n_games)]
    yAx = "Percent Wins"
    figure_file = modelName + " Validate Wins.png"
    title = "Validate Wins  for " + str(num) + " Towers "
    plot_learning_curve(x, plotRewards, title, figure_file, yAx)


    yAx = "Number of Illegal Moves"
    illegals = np.mean(illegalMoves, axis=0)
    n_games = len(illegals)
    x = [i + 1 for i in range(n_games)]
    figure_file = modelName + "Validate Illegal Average " + str(num) + ".png"
    title = modelName + "Validate Illegal " + str(num) + "best policy"
    plot_learning_curve(x, illegals, title, figure_file, yAx)