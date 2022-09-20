import random
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

latent_dim = 15  # dimension to compress to
obsLen = 38 # full dimension
# just to control writing of logs, don't worry about it
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer('log_dir')
        # self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):
        self.model = self.createModel()

        self.targetNet = self.createModel()
        self.targetNet.set_weights(self.model.get_weights())  # give target network the same weights as the model

        # deque = doubly ended queue (fast for popping and appending)
        self.memory = deque(maxlen=1000000)  # replay buffer

        # self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(modelName, int(time.time())))

        self.targetCounter = 0  # counts to tell when to update target


    def createModel(self):

        model = Sequential()
        model.add(tf.keras.Input(shape=(obsLen,)))  # input
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(128))
        # model.add(Activation('relu'))

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

        self.model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

        self.model.fit(np.array(data), np.array(Q), batch_size=batchSize, verbose=1, shuffle=False)
        if terminal:
            self.targetCounter += 1

        if self.targetCounter > updateEvery:
            self.targetNet.set_weights(self.model.get_weights())
            self.targetCounter = 0




class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
    ])

    # re-expand to 38 values
    self.decoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(obsLen, activation='linear')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



# copy of getState from HighEnvs - have to rerwrite because that one is obj oriented
def getState(tower):
    autoencoder = Autoencoder(latent_dim)
    autoencoder.load_weights("Autoencoder")
    index = 0
    ids = ["Tile01", "Tile02", "Tile03", "Tile04", "Tile05", "Tile06", "Tile07", "Tile08", "Tile09", "Tile10",
           "Tile11", "Tile12", "Tile13", "Tile14", "Tile15", "Tile16", "Tile17", "Tile18"]
    state = np.zeros(obsLen)
    floorCount = 0
    prevRot = 0
    prevOri = [0, 0, 0]  # origin of 1st tile is [0,0,0]
    for floor in tower:  # should only have 1 item in each
        for tile in floor:
            id = ids.index(tile.id)
            state[index] = id
            index += 1
            # tile.origin is with respect to the origin of the tile underneath
            tempOri = [x - y for x, y in zip(tile.origin, prevOri)]
            state[index:index + 3] = [round(i / 500, 2) for i in tempOri]
            index += 3
            # same for rotation
            temp = np.deg2rad(tile.rotation - prevRot)
            state[index] = np.sin(temp)
            index += 1
            state[index] = np.cos(temp)
            index += 1

            prevRot = tile.rotation
            prevOri = tile.origin
            count = 0  # make sure there are always 5 spots
            for spot in tile.spots:  # use local spots
                if tile.filled[count]:
                    state[index] = -1  # filled, otherwise provide colour
                else:
                    state[index] = tile.colours[count]
                index += 1
                count += 1

            for i in range(count, 5):
                state[index] = 6  # ghost spots are 6
                index += 1
        floorCount += 1

    for _ in range(floorCount, 3):
        index += 11

    # the -1 doesn't matter
    state[index:index + 5] = HighEnv1.getPillars(-1, 5)  # [x+1 for x in self.myPillars]
    index += 5

    encodedState = autoencoder.encoder(np.array([state])).numpy()
    return list(encodedState)

def trainEncoder():
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=['accuracy'])

    x_train = []
    x_test = []

    file = open("MinTowerSim_1000.txt", "rb")
    for i in range(300):
        temp = pickle.load(file)[0]
        state = getState(temp)
        x_train.append(state)
    file.close()


    file = open("TestTowerSim_100.txt", "rb")
    for i in range(25):
        temp = pickle.load(file)[0]
        state = getState(temp)
        x_test.append(state)
    file.close()

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # print(x_train)
    history = autoencoder.fit(x_train, x_train,
                    epochs=15000,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    encoded = autoencoder.encoder(x_train).numpy()
    decoded = autoencoder.decoder(encoded).numpy()
    print()
    print("Training Results")
    print()
    index = 0
    for e, d in zip(encoded, decoded):
        print([round(i,1) for i in e])
        print([round(i, 1) for i in x_train[index]])
        print([round(i,1) for i in d])
        index += 1
        print()

    print("Testing Results")
    print()
    encoded = autoencoder.encoder(x_test).numpy()
    decoded = autoencoder.decoder(encoded).numpy()
    autoencoder.save_weights("Autoencoder")

    index = 0
    for e, d in zip(encoded, decoded):
        print([round(i,1) for i in e])
        print([round(i, 1) for i in x_test[index]])
        print([round(i,1) for i in d])
        index += 1
        print()


def trainDQN():
    modelName = "DQN"
    env = HighEnv1()

    epsilon = 1
    decay = 0.99
    minEps = 0.05
    checkProgress = 150  # check progress of net every 50 steps for best model
    run= "_100NoSim"

    # loop through episodes
    agent = DQNAgent()
    episodeRewards = []
    bestReward = -np.inf
    count = 0
    for episode in tqdm(range(1, 1000), ascii=True, unit='episodes'):
        print("Episode: ", count)
        count += 1
        # agent.tensorboard.step = episode
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
                agent.model.save_weights("models/best", modelName)
                bestReward = averageReward

        # Decay epsilon
        if epsilon > minEps:
            epsilon *= decay
            epsilon = max(minEps, epsilon)

    agent.model.save_weights("models/final" + modelName)

    n_games = len(episodeRewards)
    x = [i + 1 for i in range(n_games)]
    yAx = "Rewards"
    figure_file = "models/Final " + modelName + " Rewards.png"
    title = modelName
    plot_average_curve(x, episodeRewards, title, figure_file, yAx)

    yAx = "Number of Illegal Moves"
    illegals = env.getIllegal()
    n_games = len(illegals)
    x = [i + 1 for i in range(n_games)]
    figure_file = "models/Final " + modelName + "Illegal.png"
    title = modelName + "Illegal"
    plot_average_curve(x, illegals, title, figure_file, yAx)


    print("____________TESTING____________")

    # test agent
    env = HighEnv1("TestTowerSim_100.txt")
    # agent.model = tf.keras.models.load_model('models/final',modelName,".model")
    episodeRewards = []
    # env = HighEnv1()  # can use different file for test
    num = 25
    for i in range(num):
        state = env.reset()
        done = False
        epReward = 0
        while not done:
            action = np.argmax(agent.getQ(np.array(state).reshape(-1, 38)))
            newState, reward, done, info = env.step(action)
            epReward += reward
        episodeRewards.append(epReward)


    n_games = len(episodeRewards)
    x = [i+1 for i in range(n_games)]
    yAx = "Rewards"
    figure_file = "models/Final Validate " + modelName + ".png"
    title = "Validate " + modelName + " for " + str(num) + " Towers"
    plot_learning_curve(x, episodeRewards, title, figure_file, yAx)

    yAx = "Number of Illegal Moves"
    illegals = env.getIllegal()
    n_games = len(illegals)
    x = [i + 1 for i in range(n_games)]
    figure_file = "models/Final " + modelName + "Validate Illegal.png"
    title = modelName + "Validate Illegal"
    plot_learning_curve(x, illegals, title, figure_file, yAx)


trainEncoder()
trainDQN()