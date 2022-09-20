import gym
from Utils import plot_learning_curve, plot_average_curve, saveData
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import torch as th
import pickle
from Environment import Environment
import numpy as np
import TowerSim as tower
import os
import warnings
from stable_baselines3.common.callbacks import EvalCallback

# import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore") # get rid of warnings

# to run IK uncomment next line and change run = "1024_Teleport" to prerun = "1024_Teleport"
# run = "1024_IK"
run = "1024_Teleport"
modelName = "SAC"
num = 1000
yAx = "Rewards"
ver = str(num) + "_"

fileName = "TowerModels" + str(num) + ".txt"
f = open(fileName, 'rb')
temp = pickle.load(f)
f.close()
spots = temp[0]
filled = temp[1]
origins = temp[2]
goal = temp[3]
occupied = tower.getOccupied(spots, filled)
env = Environment(spots, filled, occupied, origins, goal, fileName=fileName)
env = Monitor(env)

n_actions = env.action_space.shape[-1]
policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[1024, 1024, 512, 512])

# for IK, train on previously trained model. Uncomment line 43 and 50, comment line 45 and 49.
# model = SAC.load(modelName + "_sawyer_" + ver + prerun + "_best/best_model", env=env) # train more on pretrained
# for teleport, train from scratch
model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_starts=10, learning_rate=0.0003, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path=modelName + "_sawyer_" + ver + run + "_best",
                             log_path='./logs/', eval_freq=100,
                             deterministic=True, render=False)
model.learn(total_timesteps=250000, log_interval=50, callback=eval_callback) # for teleport
# model.learn(total_timesteps=50000, log_interval=50, callback=eval_callback) # for IK

model.save(modelName + "_sawyer_" + ver + run)
print("Saved!\n")

temp = env.get_episode_rewards()
n_games = len(temp)
x = [i+1 for i in range(n_games)]
figure_file = modelName + run + ".png"
title = modelName + " " + run
saveData(temp, run+"TrainingData.txt")

plot_average_curve(x, temp, title, figure_file, yAx)
del model # remove to demonstrate saving and loading

# can load best or end model. Best model is recommended
model = SAC.load(modelName + "_sawyer_" + ver + run + "_best/best_model")
# model = SAC.load(modelName + "_sawyer_" + ver + run)

# for testing
num = 50
fileName = "TowerModels" + str(num) + ".txt"
# number of towers in file
f = open(fileName, 'rb')
temp = pickle.load(f)
f.close()
spots = temp[0]
filled = temp[1]
origins = temp[2]
goal = temp[3]
occupied = tower.getOccupied(spots, filled)
env = Environment(spots, filled, occupied, origins, goal, fileName=fileName)
env = Monitor(env)

obs = env.reset()
score = 0
count = 0
scores = []

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    score += rewards
    if dones:
        obs = env.reset()
        count += 1
        scores.append(score)
        score = 0
    if count >= num:
        x = [i for i in range(len(scores))]
        title = "Check Test for " + str(num) + " Towers " + run + "_best policy"
        figure_file = title + ".png"
        plot_learning_curve(x, scores, title, figure_file, yAx)
        break