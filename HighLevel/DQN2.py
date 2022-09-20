import random

import keras.optimizers
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.keras.callbacks import TensorBoard
import time
from MinHighSim import Tile
import numpy as np
from HighEnvs import HighEnv1
from tqdm import tqdm
from Utils import plot_learning_curve, plot_average_curve
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import pickle
import matplotlib.pyplot as plt

startsLearning = 100  # starts training after 50 timesteps
batchSize = 64
gamma = 0.99
updateEvery = 1000  # how often to update target net

obsLen = 38

class DQNAgent:
    def __init__(self):

        # traditional model
        # self.model = self.createModel()
        # self.targetNet = self.createModel()
        # self.targetNet.set_weights(self.model.get_weights())  # give target network the same weights as the model

        # attention model
        self.model = self.attention()
        self.targetNet = self.attention()
        self.targetNet.set_weights(self.model.get_weights())

        # deque = doubly ended queue (fast for popping and appending)
        self.memory = deque(maxlen=1000000)  # replay buffer
        self.targetCounter = 0  # counts to tell when to update target


    def attention(self):
        inp = tf.keras.layers.Input(shape=(38,))
        dense = tf.keras.layers.Dense(units=128, activation='relu')(inp)
        dense = tf.keras.layers.Dense(units=128, activation='relu')(dense)
        queryEncoding = tf.keras.layers.Dense(units=64, activation='relu')(dense)
        valueEncoding = tf.keras.layers.Dense(units=64, activation='relu')(dense)

        attention = tf.keras.layers.Attention()([queryEncoding, valueEncoding])

        concat = tf.keras.layers.Concatenate()([queryEncoding, attention])

        output = tf.keras.layers.Dense(5, activation='softmax')(concat)

        model = tf.keras.Model(inp, output)

        return model

    def createModel(self):

        model = Sequential()
        model.add(tf.keras.Input(shape=(obsLen,)))  # input
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))

        model.add(Dense(5, activation='softmax'))  # output
        model.compile(loss="mse", optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        return model



    def updateMem(self, transition):
        self.memory.append(transition)  # add transition to queue

    # gets Q values for a given state
    def getQ(self, state):
        return self.model.predict(state)[0]

    # state action reward newState done?
    def train(self, terminal, step):
        if len(self.memory) < startsLearning: # don't learn yet
            return

        batch = random.sample(self.memory, batchSize)
        currentStates = np.array([i[0] for i in batch])  # extract states
        currentQList = self.model.predict(currentStates)  # get q vals for all states

        nextStates = np.array([i[3] for i in batch])
        nextQList = self.targetNet.predict(nextStates)  # update based on target net

        # training data to build from batch
        data = []
        Q = []

        for index, (state, action, reward, newState, done) in enumerate(batch):
            if not done:
                maxFutureQ = np.max(nextQList[index])  # takes max of Q for a given state
                newQ = reward + gamma*maxFutureQ
            else:
                newQ = reward

            currentQ = currentQList[index]
            currentQ[action] = newQ  # set Q value to be maximum assumed

            data.append(state)
            Q.append(currentQ)  # corrected Q


        self.model.fit(np.array(data), np.array(Q), batch_size=batchSize, verbose=1, shuffle=False)
        if terminal:
            self.targetCounter += 1

        if self.targetCounter > updateEvery:
            self.targetNet.set_weights(self.model.get_weights())
            self.targetCounter = 0


def trainDQN():
    modelName = "AttentionDQN2"
    env = HighEnv1()

    epsilon = 1
    decay = 0.99
    minEps = 0.05
    checkProgress = 150  # check progress of net every 50 steps for best model

    # if not running an attention model, change 'attn' to appropriate label
    run = "Simulate attn Average" # for simulation
    # run = "NoSim attn Average Training" # for non-simulation

    # loop through episodes
    agent = DQNAgent()
    agent.model.load_weights("models/best" + modelName)
    agent.model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    episodeRewards = []
    bestReward = -np.inf
    count = 0
    for episode in tqdm(range(1,50000), ascii=True, unit='episodes'):
        print("Episode: ", count)
        count += 1
        epReward = 0
        step = 0
        state = env.reset()
        done = False
        while not done:
            if np.random.random() > epsilon:  # do actual move
                action = np.argmax(agent.getQ(np.array(state).reshape(-1, 38)))
            else:  # random action
                action = np.random.randint(0, 4)

            newState, reward, done, info = env.step(action)
            epReward += reward
            agent.updateMem((state, action, reward, newState, done))
            agent.train(done, step)
            state = newState
            step += 1

        episodeRewards.append(epReward)
        if not episode % checkProgress or episode == 1:
            averageReward = sum(episodeRewards[-checkProgress:]) / len(episodeRewards[-checkProgress:])
            if averageReward >= bestReward:
                agent.model.save_weights("models/best" + modelName + run)
                bestReward = averageReward

        # Decay epsilon
        if epsilon > minEps:
            epsilon *= decay
            epsilon = max(minEps, epsilon)

    agent.model.save_weights("models/end" + modelName + run)

    n_games = len(episodeRewards)
    x = [i + 1 for i in range(n_games)]
    yAx = "Rewards"
    figure_file = "models/" + run + modelName + " Rewards.png"
    title = modelName
    plot_average_curve(x, episodeRewards, title, figure_file, yAx)

    yAx = "Number of Illegal Moves"
    illegals = env.getIllegal()
    n_games = len(illegals)
    x = [i + 1 for i in range(n_games)]
    figure_file = "models/" + run + modelName + "Illegal.png"
    title = modelName + "Illegal"
    plot_average_curve(x, illegals, title, figure_file, yAx)


    print("____________TESTING____________")

    # test agent
    agent = DQNAgent() # reinitialize and load model
    agent.model.load_weights("models/best" + modelName)

    totalRewards = []
    illegalMoves = []
    winArray = []
    num = 100
    numRuns = 100
    for j in range(numRuns):
        # reset env
        env = HighEnv1("TestTowerSim_100.txt") # re-initialize environment after full run through wanted test data
        episodeRewards = []
        for i in range(num):
            print("Episode:", i)
            state = env.reset()
            done = False
            epReward = 0
            while not done:
                q = agent.getQ(np.array(state).reshape(-1, 38))
                action = np.argmax(q)
                newState, reward, done, info = env.step(action)
                state = newState
                epReward += reward
            episodeRewards.append(epReward)
        totalRewards.append(episodeRewards)
        k = env.getIllegal()
        illegalMoves.append(k)
        wins = [1 if x <= 5 else 0 for x in k]
        winArray.append(wins)

    n_games = len(episodeRewards)
    plotRewards = np.mean(totalRewards, axis=0)
    print("Average reward:")
    print(np.mean(plotRewards))
    x = [i+1 for i in range(n_games)]
    yAx = "Rewards"
    figure_file = "models/" + run + modelName + " Validate Rewards.png"
    title = run + "Validate  for " + str(num) + " Towers "
    plot_learning_curve(x, plotRewards, title, figure_file, yAx)

    n_games = len(wins)
    plotWins = np.mean(winArray, axis=0)
    print("Average win:")
    print(np.mean(plotWins))
    x = [i+1 for i in range(n_games)]
    yAx = "Percent Wins"
    figure_file = "models/" + run + modelName + " Validate Wins.png"
    title = run + "Validate Wins  for " + str(num) + " Towers "
    plot_learning_curve(x, plotWins, title, figure_file, yAx)

    yAx = "Number of Illegal Moves"
    illegals = np.mean(illegalMoves, axis=0)
    n_games = len(illegals)
    x = [i + 1 for i in range(n_games)]
    figure_file = "models/" + run + modelName + " Validate Illegal.png"
    title = run + "Validate Illegal"
    plot_learning_curve(x, illegals, title, figure_file, yAx)

if __name__ == "__main__":
    trainDQN()