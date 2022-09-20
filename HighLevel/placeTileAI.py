import gym
from Utils import plot_learning_curve, plot_average_curve, saveData
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import pickle
from placeTileEnv import TileEnv
import numpy as np
import TowerSim as tower
import os
import warnings
from stable_baselines3.common.callbacks import EvalCallback

# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore")  # get rid of warnings

run = "TilePlace"
modelName = "SAC"
num = 1000
ver = str(num) + "_"
fileName = "TestTileTower.txt"
writeFile = 0
env = TileEnv(fileName, writeFile)
env = Monitor(env)


if __name__ == "__main__":
    n_actions = env.action_space.shape[-1]
    policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[128, 128])

    # model = SAC.load(modelName + "_sawyer_" + ver + run + "_best/best_model", env=env) # train more on pretrained
    model = SAC("MlpPolicy", env, learning_rate=0.0003, learning_starts=10, verbose=1)

    eval_callback = EvalCallback(env, best_model_save_path=modelName + ver + run + "_best",
                                 log_path='./logs/', eval_freq=100,
                                 deterministic=False, render=False)
    model.learn(total_timesteps=25000, log_interval=50, callback=eval_callback) # for ppo, sac
    model.save(modelName + ver + run)
    print("Saved!\n")
    temp = env.get_episode_rewards()
    saveData(temp, "PlaceTileData.txt")
    n_games = len(temp)
    x = [i+1 for i in range(n_games)]
    figure_file = modelName + run + ".png"
    yAx = "Rewards"
    title = modelName + " " + run
    plot_average_curve(x, temp, title, figure_file, yAx)

    del model # remove to demonstrate saving and loading

    model = SAC.load(modelName + ver + run + "_best/best_model")

    # for testing
    num = 50
    yAx = "Rewards"
    env = TileEnv(fileName, 1)
    env = Monitor(env)

    obs = env.reset()
    score = 0
    count = 0
    scores = []
    wins = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if rewards == 10:
            wins += 1
        score += rewards
        if dones:
            obs = env.reset()
            count += 1
            scores.append(score)
            score = 0
        if count >= num:
            x = [i for i in range(len(scores))]
            title = "Test for " + str(num) + " Towers " + run + "_best policy"
            figure_file = title + ".png"
            plot_learning_curve(x, scores, title, figure_file, yAx)
            break

    print(wins)